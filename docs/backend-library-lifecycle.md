# Backend library lifecycle

Norma persists album discovery and long-running preparation state in SQLite.
This lets a browser reconnect after a refresh or service restart without
re-indexing an album just to reconstruct the UI.

## Album catalog

```text
GET /albums?limit=50&offset=0
GET /albums/{album_id}
GET /albums/{album_id}/photos?limit=100&offset=0&sort=quality&include_rejects=true
GET /albums/{album_id}/people
GET /albums/{album_id}/selections?limit=50&offset=0
```

Album summaries report total photos, quality-complete photos, rejects,
embeddings, people-processed photos, detected faces, and selections. The
quality and people processed counts let the website distinguish not-run,
partial, and ready module states without a separate in-memory flag. Photo pages
can sort by path, quality, or capture time. Pagination bounds are validated by
FastAPI and sorting is restricted to an allow-list. The people `GET` endpoint
rehydrates persisted face clusters without running detection again.

The summary's `people_provider` is populated only when the fresh, processed
photos have one unique face-provider fingerprint. `/health` separately exposes
the worker's active `face_provider`. The website considers people analysis
ready, and calls the people `GET` endpoint automatically, only when every photo
is fresh and the two provider values match. Thus a saved Haar/DCT result or an
older model/alignment/clustering revision is shown as needing a new run instead
of being silently reused.

## Background preparation

`POST /jobs/prepare` drives both the fast base import and the three opt-in
analysis modules. The browser sends all flags explicitly. A base import uses:

```powershell
$job = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/jobs/prepare `
  -ContentType application/json `
  -Body '{"folder":"D:\\Photos\\Trip","include_quality":false,"include_embeddings":false,"include_people":false}'

Invoke-RestMethod http://127.0.0.1:8765/jobs/$($job.id)
```

After the album exists, submit the same folder with exactly one module enabled:

| Operation | `include_quality` | `include_embeddings` | `include_people` |
| --- | ---: | ---: | ---: |
| Quality signals + similarity groups | `true` | `false` | `false` |
| Semantic embedding index | `false` | `true` | `false` |
| Face detection + people grouping | `false` | `false` | `true` |

The default people provider is OpenCV YuNet 2023mar plus SFace. YuNet detects
on a preview capped at 1600 pixels on its longest side with a score threshold
of 0.8. SFace uses the five returned landmarks with `alignCrop`, and its
128-dimensional descriptor is L2-normalized. The provider fingerprint includes
the pinned model SHA values, alignment revision, and clustering-policy revision.

On the first people request, the two ONNX files (about 37 MB combined) are
downloaded to a unique temporary file, verified against fixed SHA-256 values,
and atomically moved into the local model cache. Later jobs reuse those files.
Only the public model weights are downloaded; photos and all derived data stay
on the local machine. OpenCV Zoo licenses the
[YuNet files under MIT](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE)
and the
[SFace files under Apache-2.0](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE).
The old Haar/DCT path is available only as an explicit lower-quality fallback:

```powershell
$env:NORMA_FACE_PROVIDER = "opencv-haar"
python -m ai web
```

Quality scoring and both perceptual hashes deliberately share one decoded-image
pass. Similarity grouping is an album-wide result and is exposed only when the
current photos have complete hashes. Existing API clients remain compatible:
omitted `include_quality`, `include_embeddings`, and `include_people` flags all
default to `true`, preserving the historical full preparation pipeline.

Job APIs:

```text
POST /jobs/prepare
GET  /jobs?status=running&limit=50&offset=0
GET  /jobs/{job_id}
POST /jobs/{job_id}/cancel
```

The manager intentionally runs one preparation job at a time so multiple large
albums do not compete for the machine. Inside the indexing stage, one job uses
at most four bounded photo-scan workers and keeps at most eight futures in
flight. SQLite persistence and job progress updates stay on the orchestration
thread. Additional album submissions remain `queued`.

State transitions are persisted. Bracketed stages run only when their request
flag is enabled:

```text
queued -> running/indexing -> [running/embedding] -> [running/people] -> completed
                    |                 |                    |
                    +-----------------+--------------------+-> cancelled
                    (any running stage can also become failed)
```

For base import and quality-only requests, the indexing stage owns the complete
0–100% range. Requests for semantic or people analysis include a short source
validation/indexing portion before their requested stage. Progress ranges are
allocated only to stages selected by the request, remain monotonic, and always
finish at 100%. Each active stage also stores compact `completed` / `total`
counts, so the percentage is backed by processed work rather than an animation.

Cancellation is cooperative at base/quality photo boundaries, embedding chunks,
people-stage photo boundaries, and between stages. It never interrupts a photo
while the source is being read and never persists a partial album snapshot.
If the process exits during a running job, the next startup records that job as
`failed/interrupted`; queued jobs are scheduled again. Valid chunks remain
reusable, so a new prepare job computes only the missing photos.

SQLite schema v9 allows the same source path to have album-scoped photo IDs in
overlapping parent/child albums while preserving legacy IDs during incremental
refresh. It also fingerprints indexed sources with file size and nanosecond
modification time. A base import creates metadata and thumbnails for new or
changed files while leaving their optional analysis fields empty. An unchanged
re-index reuses valid thumbnails and any completed analysis. Embedding also
requires the active provider and its recorded source fingerprint to match.
People analysis separately records provider, source fingerprint, processed
state, and face count, so unchanged photos—including zero-face photos—reuse
detection while clusters are rebuilt from the combined descriptors. Rebuilding
uses deterministic two-stage constrained agglomeration. A strict seed pass
checks cross-cluster minimum, mean, and centroid similarity. A prototype pass
then joins only mutual best candidates that pass size-aware centroid, mean, and
strongest-pair gates. Faces from the same photo remain a hard cannot-link in
both passes. Stage summaries expose `computed_count` and `reused_count` for
auditing. The prototype pass is an experimental organizer default; selecting
`NORMA_FACE_PROVIDER=opencv-yunet-sface-strict` retains YuNet/SFace while using
only the strict seed stage.

Duplicate active jobs for the same resolved folder are rejected with HTTP 409.
Completed results contain compact stage summaries rather than thousands of
photo records; clients retrieve photos through the paginated catalog.
