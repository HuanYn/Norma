# Backend library lifecycle

Norma persists album discovery and long-running preparation state in SQLite.
This lets a browser reconnect after a refresh or service restart without
re-indexing an album just to reconstruct the UI.

## Album catalog

```text
GET /albums?limit=50&offset=0
GET /albums/{album_id}
GET /albums/{album_id}/photos?limit=100&offset=0&sort=quality&include_rejects=true
GET /albums/{album_id}/selections?limit=50&offset=0
```

Album summaries report photo, reject, embedding, face, and selection counts.
Photo pages can sort by path, quality, or capture time. Pagination bounds are
validated by FastAPI and sorting is restricted to an allow-list.

## Background preparation

Submit an index + semantic embedding + optional people-grouping pipeline:

```powershell
$job = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/jobs/prepare `
  -ContentType application/json `
  -Body '{"folder":"D:\\Photos\\Trip","include_people":true}'

Invoke-RestMethod http://127.0.0.1:8765/jobs/$($job.id)
```

Job APIs:

```text
POST /jobs/prepare
GET  /jobs?status=running&limit=50&offset=0
GET  /jobs/{job_id}
POST /jobs/{job_id}/cancel
```

The v1 manager intentionally runs one preparation job at a time. Photo decoding
and embedding are CPU-heavy, and serial execution avoids competing jobs making
a local machine unresponsive. Additional submissions remain `queued`.

State transitions are persisted:

```text
queued -> running/indexing -> running/embedding -> running/people -> completed
                              \-> cancelled (between stages)
                              \-> failed
```

Cancellation is cooperative between stages and embedding chunks. It never
interrupts a photo while the source is being read. Embedding-stage progress
stores compact `completed` / `total` counts and advances between 55% and 80%.
If the process exits during a running job, the next startup records that job as
`failed/interrupted`; queued jobs are scheduled again. Valid chunks remain
reusable, so a new prepare job computes only the missing photos.

SQLite schema v5 fingerprints indexed sources with file size and nanosecond
modification time. An unchanged re-index reuses analysis/thumbnails; embedding
also requires the active provider and its recorded source fingerprint to match.
Stage summaries expose `computed_count` and `reused_count` for auditing.

Duplicate active jobs for the same resolved folder are rejected with HTTP 409.
Completed results contain compact stage summaries rather than thousands of
photo records; clients retrieve photos through the paginated catalog.
