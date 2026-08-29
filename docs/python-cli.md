# Norma local Python CLI

Norma's automation entry point is `python -m ai`. Every command uses the same
local SQLite database and derived caches as the website. Original JPG/JPEG files
remain read-only.

## Install

```powershell
python -m pip install -e ".[dev,selection,multimodal]"
```

`selection` installs OR-Tools CP-SAT. Omitting it keeps the deterministic exact
fallback for the currently supported cardinality and similarity-group limits.
`multimodal` installs the multilingual OpenCLIP runtime used by the default
learned image/text embedding backend. The model is still loaded lazily, only
when semantic indexing or an explicit warmup starts. For a zero-download
handcrafted comparison baseline, set
`NORMA_EMBEDDING_PROVIDER=lightweight` before starting the command.

By default state is stored in `.norma/data`. Override it for any command by
placing `--data-dir` before the subcommand:

```powershell
python -m ai --data-dir D:\NormaData --pretty init
```

Inspect provider availability and the active versioned cache identity:

```powershell
python -m ai --pretty providers
python -m ai --pretty provider-status
python -m ai --pretty warmup
```

See [multimodal-provider.md](multimodal-provider.md) for model cache and device
configuration.

## Album pipeline

```powershell
# One step: read-only index + semantic embedding + people grouping
python -m ai --pretty prepare "D:\Photos\Trip"

# Faster preparation without face detection
python -m ai --pretty prepare "D:\Photos\Trip" --skip-people

# Individual stages
python -m ai --pretty index "D:\Photos\Trip"
python -m ai --pretty embed ALBUM_ID
python -m ai --pretty people ALBUM_ID

# Inspect IDs and cached status
python -m ai --pretty albums
python -m ai --pretty album ALBUM_ID
python -m ai --pretty photos ALBUM_ID
python -m ai --pretty photos ALBUM_ID --include-rejects
```

`index`, `embed`, and `prepare` report `computed_count` and `reused_count`.
Repeating `prepare` on an unchanged folder reuses thumbnails, analysis, and
semantic vectors. If an embedding run fails or is cancelled between chunks, the
next run resumes from the committed photos.

## Retrieval and collection selection

```powershell
python -m ai --pretty search ALBUM_ID "夜景 blue city" --limit 20
python -m ai --pretty image-search ALBUM_ID PHOTO_ID --limit 20
python -m ai --pretty select ALBUM_ID "选 12 张夜景，质量至少 45，相似组最多 1 张"
```

All output is UTF-8 JSON. Domain failures also return JSON on stderr and a
non-zero process exit code.

## Preference, replacement, and audit

```powershell
python -m ai --pretty feedback ALBUM_ID PREFERRED_PHOTO_ID REJECTED_PHOTO_ID `
  --selection-id SELECTION_ID

python -m ai --pretty replace SELECTION_ID REMOVE_PHOTO_ID
python -m ai --pretty show-selection SELECTION_ID
python -m ai --pretty show-preferences --user-id local
python -m ai --pretty selection-history ALBUM_ID
python -m ai --pretty jobs --status completed
python -m ai --pretty show-job JOB_ID
```

Replacement locks every non-removed photo and returns infeasible rather than
silently relaxing the original hard constraints.

## Human relevance evaluation

```powershell
python -m ai --pretty eval-add-query ALBUM_ID "城市夜景" `
  --notes "人工评估查询"
python -m ai --pretty eval-queries ALBUM_ID
python -m ai --pretty eval-candidates QUERY_ID --limit 50
python -m ai --pretty eval-judge QUERY_ID PHOTO_ID 3 --annotator local
python -m ai --pretty eval-run ALBUM_ID --cutoffs 1 5 10 20
python -m ai --pretty eval-show-run RUN_ID
```

Relevance uses four grades: `0` not relevant, `1` marginal, `2` relevant, and
`3` highly relevant. Label the retrieved pool as well as known misses whenever
possible. Unjudged photos count as non-relevant in a run, so sparse labels can
underestimate recall. Each run persists its rankings and label snapshot for
later audit.

## Cache maintenance

```powershell
# Default: report old unreferenced generated files without deleting anything
python -m ai --pretty cache-gc

# Review the report, then explicitly apply with the same age boundary
python -m ai --pretty cache-gc --apply --min-age-seconds 3600
python -m ai --pretty cache-usage
python -m ai --pretty cache-enforce --budget-gb 20
python -m ai --pretty cache-enforce --budget-gb 20 --apply
python -m ai --pretty maintenance-runs --limit 20
```

The collector only scans derived thumbnail, embedding, and face roots under the
selected `--data-dir`. It never scans the model cache or source photo folders.
Apply mode is blocked while prepare jobs are queued or running. Keep a nonzero
age boundary when direct synchronous API work may be running.

## Website and API process

Run the complete local website:

```powershell
python -m ai --data-dir D:\NormaData web --host 127.0.0.1 --port 8765
```

Run the same FastAPI application under the API-oriented command name:

```powershell
python -m ai --data-dir D:\NormaData serve --host 127.0.0.1 --port 8765
```

Both commands serve the API; `web` additionally checks that compiled Vue assets
exist. The album and AI commands above run domain code directly in the current
Python process.
