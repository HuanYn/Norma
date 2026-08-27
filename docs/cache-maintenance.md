# Generated cache maintenance and provider warmup

## Safe cache collection

Norma treats thumbnails, photo embeddings, face descriptors, and face crops as
disposable derivatives. SQLite is the authority for which generated files are
still live.

```text
POST /maintenance/cache/gc
```

The request defaults to:

```json
{"dry_run": true, "min_age_seconds": 3600}
```

The service scans exactly three resolved roots under the active data directory:
`thumbnails/`, `embeddings/`, and `faces/`. It does not scan `models/`, the
SQLite database, web assets, or source albums. A resolved path escaping its root
is counted as unsafe and never deleted.

The keep set contains every `photos.thumbnail_path`, `photos.embedding_path`,
`faces.embedding_path`, and the deterministic crop path for every persisted
face. Unreferenced files younger than the age gate are reported separately and
retained. Apply mode refuses to run while a queued/running prepare job exists.

Dry-run and apply return scanned/reference/orphan/deleted counts and bytes, up
to 20 orphan samples, unsafe counts, and bounded error details. Empty folders
are deliberately retained.

## Warmup state machine

`GET /providers/embedding/status` is side-effect free. It reports provider,
dimension, model-backed flag, load state, device, warmup state, timestamps, and
the last failure.

`POST /providers/embedding/warmup` returns HTTP 202 immediately. One daemon
thread calls the provider probe; concurrent/repeated requests are idempotent.
The active provider instance comes from the same process cache used by search,
selection, feedback, and embedding, so a successful warmup removes their first
model-load latency.

Warmup does not download silently in offline mode: missing or incompatible
model state becomes `failed` with an explicit error. It does not create image
embeddings or modify albums.

## Usage, budget, and audit history

```text
GET  /maintenance/cache/usage
POST /maintenance/cache/enforce
GET  /maintenance/runs?limit=50&offset=0
```

Usage reports separate file/byte totals for thumbnails, embeddings, faces, and
models, plus SQLite database/WAL/SHM bytes. `total_state_bytes` is compared with
an optional `NORMA_CACHE_BUDGET_GB` value.

Quota enforcement remains dry-run by default. A request may override the budget
with exact `budget_bytes`. It runs the same age-gated orphan analysis and
returns current and projected totals. If referenced caches, models, and the
database alone exceed the budget, the response is unsatisfied with an explicit
warning; Norma does not invent an eviction policy or delete live data.

Every explicit GC/quota attempt, including failures caused by active jobs or a
missing budget, is written to schema v8 `maintenance_runs`. History is bounded
and paginated. Usage reads are not persisted because they are side-effect-free
observations.
