# Norma architecture

Norma is a local website backed by Python. FastAPI serves both the compiled Vue
application and the domain API, so browser requests, derived media, and SQLite
state remain on one machine and one origin. The Python CLI calls the same domain
services directly for automation.

## Runtime boundary

```text
Browser (Vue)
      │ same-origin HTTP
      ▼
FastAPI ───────────────► Python domain services
      │                         │
      │                         ├── SQLite metadata and feedback
      │                         └── thumbnails / embeddings / face crops
      ▼
compiled web assets                 (never the source folder)
```

`python -m ai web` starts the complete product. `pnpm dev` is only the web
development server and proxies API/media routes to `python -m ai serve`.

## Data ownership

- Source JPG/JPEG files are read-only inputs.
- Source photos and derived thumbnails, face crops, and descriptors are never
  uploaded. A first face run may download only the public model files described
  below.
- The initial folder open records file inventory/basic metadata and creates
  thumbnails; quality, embeddings, and faces are opt-in follow-up work.
- Thumbnails, embeddings, face crops, and SQLite live under `.norma/` by default.
- Generated caches are disposable; source paths are revalidated during indexing.
- No vector database is needed for album-sized data. Normalized NumPy embeddings
  are cached and searched with exact cosine similarity.

## Delivered milestones

1. **M0:** local Python service, SQLite schema, health/capability contract, web shell.
2. **M1:** fast JPG catalog/thumbnails plus on-demand quality signals, similarity groups, and reject fold.
3. **M2:** semantic embeddings, people clusters, text/image retrieval.
4. **M3:** structured constraints, auditable scoring, CP-SAT optimization.
5. **M4:** grounded reasons, locked replacement, pairwise preference learning.
6. **M5:** persistent album catalog, selection history, and queued preparation jobs.
7. **M6:** optional multilingual OpenCLIP, batch inference, and strict provider-scoped caches.
8. **M7:** incremental indexing, source-fingerprinted vectors, chunk recovery, and job progress.
9. **M8:** persistent relevance judgments and provider-scoped retrieval evaluation.
10. **M9:** source-fingerprinted incremental people detection and cluster rebuilds.
11. **M10:** reference-aware cache collection and asynchronous provider warmup.
12. **M11:** disk usage accounting, quota policy, and persisted maintenance audits.

## Library lifecycle

`ai/library.py` exposes persisted album summaries, bounded photo pagination, and
selection history. Clients reconstruct their workspace after refresh from
SQLite instead of relying on one indexing response remaining in memory.

`ai/jobs.py` orchestrates the base import and opt-in quality, embedding, and
people stages through a single-worker executor. The website uses the historical
`/jobs/prepare` route with explicit `include_quality`, `include_embeddings`, and
`include_people` flags; all three default to `true` for older API clients. Job
state and compact intermediate results are persisted in SQLite. Duplicate
active folders are rejected atomically. Running work is marked interrupted
after an unclean restart, queued work is scheduled again, and cancellation is
honored at base/quality photo boundaries, embedding chunks, people-photo
boundaries, and between stages. Progress is allocated across only the requested
stages and persists both a whole-job percentage and current-stage completed and
total counts, allowing the browser to recover the active button after refresh.

Video and world generation remain deferred until the personalized selection
loop has stronger model providers and product evidence.

## Maintenance and provider runtime

`ai/maintenance/` scans only the generated `thumbnails`, `embeddings`, and
`faces` roots. SQLite paths and inferred face-crop paths form the keep set.
Model caches and source folders are outside the scan boundary. Collection is a
dry-run by default, ignores young orphans for one hour by default, and refuses
deletion while prepare jobs are queued or running.

`ai/provider_runtime.py` owns one idempotent background warmup state machine for
the active embedding provider. Status reads never load a model. A warmup request
returns immediately and moves through `idle -> loading -> ready|failed`; a
repeated request while loading does not create another model load. Optional
startup prewarming uses the same path.

SQLite schema v8 persists each explicit GC or quota request as a maintenance
run with operation, dry-run flag, request, result/error, status, and timestamps.
Usage accounting separates generated thumbnails/embeddings/faces, model files,
and SQLite state. Quota enforcement can only invoke the orphan collector; it
never evicts referenced caches or models automatically, and reports when the
budget is impossible under that safety boundary.

## Library fast path and fallback

`ai/index/` provides the active `pillow-opencv-fallback-v1` implementation. It:

- recursively discovers JPG/JPEG files without a hard album-size limit;
- scans with at most four bounded workers while keeping database writes and
  progress callbacks on the orchestration thread;
- verifies source size and modification time before and after reads;
- supports a base mode that creates metadata/thumbnails without quality or hash
  analysis, keeping the first folder-open path independent of optional models;
- combines quality signals, suggested rejects, and both perceptual hashes in one
  decoded-image pass when **质量与相似** is requested;
- reuses stored analysis and thumbnails when both source values are unchanged;
- uses album-scoped IDs for new photos, so overlapping parent/child albums can
  reference the same source path independently; schema v9 preserves legacy IDs;
- writes thumbnails only under the Norma data directory;
- computes conservative quality suggestions and never deletes a source;
- assigns similarity groups from perceptual hashes.

The vendored `crates/pianke-core` retains its MIT notice and provides a possible
future native fast path for hashing and quality primitives. It is not part of
the supported web runtime.

## Retrieval provider

`ai/index/embedding.py` defines a shared image/text embedding interface. The
default `lightweight-semantic-v1` provider produces deterministic normalized
16-dimensional CPU descriptors for color, luminance, composition, and coarse
visual concepts. It is an interpretable integration baseline.

`ai/index/openclip_provider.py` optionally loads the multilingual XLM-R +
ViT-B/32 OpenCLIP model on first inference. It batches image reads, supports CPU
or CUDA selection, and validates all 512-dimensional normalized outputs before
they enter the cache. Model import or load failures remain explicit and map to
HTTP 503 at the API boundary.

`ai/retrieval/` owns provider-scoped cache generation and cosine search. SQLite
schema v5 stores the versioned provider identity and source size/mtime used for
every photo vector. Stale photos are embedded in provider-sized chunks, each
committed independently so a retry continues after the last completed chunk.
Text, reference-photo, selection, and replacement paths reject incomplete,
stale, or mixed-provider albums instead of comparing incompatible vector
spaces. SQLite schema v6 additionally stores evaluation queries, 0..3 relevance
judgments, and immutable evaluation run reports. Schema v7 adds per-photo face
provider, source fingerprint, processed marker, and face count.

## Retrieval evaluation

`ai/evaluation/` turns model quality checks into a persistent workflow. A query
belongs to one album, each photo can receive one current 0..3 judgment per
query, and candidate ranking uses the same `RetrievalService` as production
search. Therefore stale source files, incomplete caches, and provider mismatch
fail evaluation through the same safety gate.

Runs compute binary Precision/Recall and MRR from grades greater than zero, plus
graded nDCG using the full 0..3 labels. Reports preserve the provider identity,
cutoffs, ranked photo IDs, judgment snapshot, per-query metrics, macro metrics,
and creation time in SQLite. Unjudged candidates are treated as non-relevant;
queries with no judgments are counted as skipped instead of silently entering
the macro average.

## People provider

`ai/people/` separates face detection/description from persistence and
clustering. The default `opencv-yunet-sface` path runs the OpenCV Zoo YuNet
2023mar detector on a preview capped at 1600 pixels on its longest side, using
a 0.8 detection-score threshold. Detection coordinates and five landmarks are
mapped back to the original orientation-corrected image. OpenCV SFace then
applies five-point `alignCrop`, emits a 128-dimensional descriptor, and the
provider validates and L2-normalizes it.

The YuNet and SFace ONNX files are not bundled. The first people-analysis click
downloads about 37 MB of pinned public weights into the local model cache. A
file is streamed to a unique temporary path, checked against its fixed SHA-256,
flushed, and atomically renamed; an invalid or interrupted download is never
accepted as a model. Future runs reuse the cache. The model request contains no
photo data. OpenCV Zoo distributes the
[YuNet files under MIT](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE)
and the
[SFace files under Apache-2.0](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE).

Clustering is deterministic and two-stage rather than single-link union. The
first pass builds strict seeds: candidate pairs are ordered by similarity with
stable IDs as tie breakers, and a merge must pass cross-cluster minimum, mean,
and centroid gates. The second pass repairs pose fragmentation only for mutual
best cluster candidates. Multi-member seeds and singleton attachments use
separate centroid, cross-pair mean, and strongest-pair thresholds; all evidence
is recomputed after every accepted join. Both passes reject any merge that would
place two faces from the same photo in one cluster (a hard cannot-link). This
blocks weak bridges while allowing a stable cluster prototype to recover some
low-pose-similarity faces. Uncertain single-face clusters remain visible instead
of being forced into a larger group; all groups are organizational suggestions
rather than biometric identity claims. Prototype attachment is a versioned
experimental default; `opencv-yunet-sface-strict` keeps the same detector and
recognizer but disables the second pass when false merges are especially costly.

The provider fingerprint records the YuNet and SFace model SHA prefixes, the
alignment revision, and the clustering-policy revision. SQLite and generated
paths are provider-scoped. The album catalog returns the unique provider only
for source-fingerprint-current face results, while `/health` reports the active
worker provider. The browser loads a saved people snapshot only when every
photo is current and those fingerprints match. A previous Haar/DCT result or a
result from any older model, alignment, or clustering revision therefore
requires a new people run.

`opencv-haar` remains available only as an explicit lower-quality fallback via
`NORMA_FACE_PROVIDER=opencv-haar`; it is no longer the default.

Re-indexing preserves semantic and face descriptor records for unchanged photos
and invalidates only changed or removed photos. A source-fingerprint-current
photo reuses its detection result, including an explicit zero-face result, when
provider, recorded face count, descriptor files, and crop files all validate.
After any local recomputation, clustering is rebuilt over both reused and new
descriptors so cluster membership never mixes old and new topology. Generated
files are disposable and SQLite never intentionally retains references to
invalid caches.

## Structured selection

`ai/selection/parser.py` maps a bounded bilingual prompt grammar to explicit
hard constraints: exact count, quality floor, reject policy, and maximum photos
per similarity group. Unsupported requested concepts fail visibly. Only prompts
without a semantic request may fall back to quality-only ranking.

`ai/selection/service.py` combines semantic similarity and normalized quality
into an inspectable score. `optimizer.py` enforces collection constraints. With
the optional `selection` dependency it uses deterministic OR-Tools CP-SAT with a
two-second limit. The fallback is exact for the current cardinality and
per-group-capacity family. Infeasible requests return no partial selection.

## Preference and replacement

`ai/preferences/` represents photos with bounded semantic, quality, sharpness,
brightness, contrast, and orientation features. Pairwise feedback applies an
online logistic update with decay, regularization, and clipped weights. Events
and the local model are persisted in SQLite and never sent off-device.

After comparisons exist, selection reserves 13% of its soft score for learned
preference while preserving all hard constraints. Explanations expose the
semantic, quality, constraint, and preference evidence.

Replacement locks every non-removed selected photo, then chooses the highest
scoring unselected candidate that preserves the original constraints. A success
creates a new immutable audit; failure returns infeasible without silently
changing the collection.
