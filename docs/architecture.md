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
      │                         ├── SQLite schema v14 + immutable audits
      │                         ├── thumbnails / embeddings / face crops
      │                         └── frozen local learned models
      │                              ├── OpenCLIP / YuNet / SFace
      │                              └── pinned Qwen3-VL (optional RAG)
      ▼
compiled web assets            read-only source JPG/JPEG folder
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
- Schema v13 binds each semantic vector to the full SHA-256 of the source bytes
  used to create it. Size/mtime remain the fast change detector; the content
  digest closes same-size/restored-mtime replacement cases.
- No vector database is needed for album-sized data. Normalized NumPy embeddings
  are cached and searched with exact cosine similarity.

## Delivered milestones

1. **M0:** local Python service, SQLite schema, health/capability contract, web shell.
2. **M1:** fast JPG catalog/thumbnails plus on-demand quality signals, similarity groups, and reject fold.
3. **M2:** semantic embeddings, people clusters, text/image retrieval.
4. **M3:** structured constraints, auditable scoring, CP-SAT optimization.
5. **M4:** grounded reasons, locked replacement, pairwise preference learning.
6. **M5:** persistent album catalog, selection history, and queued preparation jobs.
7. **M6:** default multilingual OpenCLIP, batch inference, and strict provider-scoped caches; the handcrafted provider remains an explicit baseline.
8. **M7:** incremental indexing, source-fingerprinted vectors, chunk recovery, and job progress.
9. **M8:** persistent relevance judgments and provider-scoped retrieval evaluation.
10. **M9:** source-fingerprinted incremental people detection and cluster rebuilds.
11. **M10:** reference-aware cache collection and asynchronous provider warmup.
12. **M11:** disk usage accounting, quota policy, and persisted maintenance audits.
13. **M12:** 67D Bayesian contextual preference learning with immutable events and versioned posterior snapshots.
14. **M13:** CAPU-PDRR-MC decision-aware pair acquisition with one-shot suggestion consumption.
15. **M14:** grounded OpenCLIP-to-local-Qwen3-VL RAG, immutable run audits, and schema-v13 source-content binding.
16. **M15:** rowid-free RAG audit storage plus pinned model/runtime/backend identities for replayable learned inference.

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

The schema-v8 migration introduced persistence for each explicit GC or quota
request as a maintenance
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
default provider is the frozen multilingual XLM-R + ViT-B/32 OpenCLIP model;
raw query text goes directly to its multilingual text tower. The deterministic
`lightweight-semantic-v1` 16-dimensional descriptor remains an explicit
integration baseline and zero-download diagnostic, never a silent fallback.

`ai/index/openclip_provider.py` lazily loads OpenCLIP on first inference. It
batches image reads, supports CPU
or CUDA selection, and validates all 512-dimensional normalized outputs before
they enter the cache. The former bounded Chinese keyword bridge is exposed only
through the separately versioned `openclip-legacy-bridge` ablation. Model
import or load failures remain explicit and map to HTTP 503 at the API boundary.

`ai/retrieval/` owns provider-scoped cache generation and exact similarity
search. The schema-v5 migration introduced versioned provider identity and
source size/mtime for each vector. Schema v13 adds
`embedding_source_sha256`: the embedding worker hashes the source before and
after inference and commits the vector, provider, source stat, and content
digest together with a compare-and-swap guard. A catalog rescan invalidates an
embedding when the bytes change even if size and mtime were restored. Existing
rows without a digest are stale and are recomputed once. Stale photos are
embedded in provider-sized chunks, each committed independently so a retry
continues after the last completed chunk.
Text, reference-photo, selection, and replacement paths reject incomplete,
stale, or mixed-provider albums instead of comparing incompatible vector
spaces. Interactive search uses the persisted binding instead of re-hashing an
entire album; the grounded RAG boundary re-hashes only its selected Top-K source
files. The schema-v6 migration additionally stores evaluation queries, 0..3
relevance judgments, and immutable evaluation run reports. Schema v7 adds
per-photo face provider, source fingerprint, processed marker, and face count.

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

For a semantic request under the default OpenCLIP provider,
`ai/selection/service.py` ranks candidates by the contextual utility described
below. Minimum quality, reject handling, exact count, and per-similarity-group
capacity remain explicit constraints rather than learned penalties. The legacy
fixed-weight score is retained only for non-contextual providers and the
quality-only compatibility path. `optimizer.py` enforces the collection
constraints. With the optional `selection` dependency it uses deterministic
OR-Tools CP-SAT with a two-second limit. The fallback is exact for the current
cardinality and per-group-capacity family. Infeasible requests return no partial
selection.

## Preference and replacement

The production OpenCLIP path uses a compact Bayesian contextual adapter rather
than the old fixed 7-feature/13%-weight compatibility model. For normalized
512D image and query vectors `z` and `q`, a fixed 32-row signed-Hadamard
projection `P` produces

```text
phi(z, q) = [Pz, (Pz) * (Pq), dot(z, q), auto_reject, quality_missing] in R^67
utility(z, q) = dot(z, q) + posterior_mean · phi(z, q)
P(i preferred to j) = sigmoid(utility_i - utility_j)
```

Training uses a zero-mean Gaussian prior, full-batch damped Newton MAP with
Armijo line search, and the inverse Hessian as a Laplace covariance. With no
compatible feedback, the returned utility is exactly normalized OpenCLIP
cosine. Search, semantic selection, replacement, and RAG retrieval all load one
compatible posterior snapshot and expose the learned residual separately from
cosine. Provider fingerprint, 67D feature schema, projection ID, comparison
count, and training-event digest are bound into the decision audit. A mismatch
fails closed or explicitly falls back to zero-feedback cosine; incompatible
spaces are never blended.

Each comparison is appended to immutable `preference_events`. Retraining writes
a new immutable `preference_models` posterior row; only deactivating the former
active row is permitted. The source event digest is rechecked when a posterior
is loaded. `tie`, `skip`, and `both_bad` remain auditable events but do not train
the binary preferred/rejected posterior. The mutable legacy model and audit are
kept for backward compatibility, not as the documented OpenCLIP decision path.

`ai/preferences/acquisition.py` implements CAPU-PDRR-MC. The production default
draws 64 posterior samples, shortlists 16 unseen pairs by predictive entropy
times posterior set-membership variance, and estimates the two possible
outcomes' posterior decision-regret reduction. Each outcome exactly re-solves
the current exact-count/partition-capacity action; posterior integration remains
Monte Carlo. Low effective sample size uses a Laplace fallback, and a violated
value-of-information invariant triggers one higher-sample retry before an
explicit abstention. Suggestions persist candidate/source digests, a digest of
the full candidate 67D decision-feature snapshot, and the displayed pair's exact
67D vectors. They exclude previously shown pairs in the same compatible
selection context and may be consumed by at most one feedback event even under
concurrent requests.

The controlled public-image experiment supports only a narrow, exploratory
claim: at 10 pair queries CAPU-PDRR-MC improved held-out pair loss, accuracy, and
constrained set regret per photo over both random and predictive-entropy
acquisition, with paired seed-bootstrap intervals in the favorable direction;
at 30 and 60 queries every corresponding interval crossed zero. See
[the full protocol and limitations](../figures/PDRR_ACQUISITION_CONTROLLED_REPORT.md).

Replacement locks every non-removed selected photo, then chooses the highest
scoring unselected candidate that preserves the original constraints. It
recomputes every locked and candidate score against one current posterior
snapshot when the model changed after the original selection. A success creates
a new immutable audit; failure returns infeasible without silently changing the
collection.

## Grounded multimodal RAG

`POST /albums/{album_id}/rag` composes the learned ranking path with a local
vision-language model. Retrieval freezes the provider, user, preference model,
training-event digest, candidate decision inputs, every embedding digest, and
the schema-v13 source-content binding. Only the bounded Top-K originals are
re-hashed and retained as immutable in-memory byte snapshots. A pinned local
`Qwen/Qwen3-VL-2B-Instruct` may generate only `claims` and `citations`; the
server constructs the canonical answer and authoritative provenance.

The generation fingerprint includes the pinned model/runtime/preprocess contract
and a digest of the system prompt, serialization, redaction, and strict output
schema. The boundary rejects duplicate JSON keys, unknown photo IDs, uncited or
duplicate claims, local-path disclosure, source/embedding/model drift, and
manifest changes. Successful `rag_runs` are immutable; schema v14 stores the
table `WITHOUT ROWID`, closing SQLite replace-by-rowid overwrite semantics in
addition to update/delete/duplicate-ID triggers. The validation level is
referential citation and provenance integrity only: there is no semantic
entailment verifier. OpenCLIP and Qwen3-VL remain frozen, so this path is not
DPO, SFT, LoRA, or end-to-end multimodal fine-tuning. Installation, request
examples, model pinning, and resource limits are in
[grounded-multimodal-rag.md](grounded-multimodal-rag.md).
