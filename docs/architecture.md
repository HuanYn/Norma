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
- Thumbnails, embeddings, face crops, and SQLite live under `.norma/` by default.
- Generated caches are disposable; source paths are revalidated during indexing.
- No vector database is needed for album-sized data. Normalized NumPy embeddings
  are cached and searched with exact cosine similarity.

## Delivered milestones

1. **M0:** local Python service, SQLite schema, health/capability contract, web shell.
2. **M1:** JPG scan, thumbnails, quality signals, similarity groups, reject fold.
3. **M2:** semantic embeddings, people clusters, text/image retrieval.
4. **M3:** structured constraints, auditable scoring, CP-SAT optimization.
5. **M4:** grounded reasons, locked replacement, pairwise preference learning.

Video and world generation remain deferred until the personalized selection
loop has stronger model providers and product evidence.

## Library fast path and fallback

`ai/index/` provides the active `pillow-opencv-fallback-v1` implementation. It:

- recursively discovers JPG/JPEG files without a hard album-size limit;
- verifies source size and modification time before and after reads;
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
visual concepts. It is an interpretable integration baseline, not a claim of
CLIP-level understanding.

`ai/retrieval/` owns provider-scoped cache generation and cosine search. Text,
reference-photo, and candidate-subset requests use one result schema, allowing a
future CLIP/SigLIP provider without changes to the web contract.

## People provider

`ai/people/` separates face detection/description from persistence and
clustering. `opencv-haar-dct-v1` stores a 79-dimensional DCT/color descriptor
and uses a conservative `0.985` similarity threshold. Single-face clusters stay
visible instead of being forced into an identity group. This is pipeline
scaffolding, not biometric identification.

Re-indexing invalidates stale semantic and people records. Generated files are
disposable and SQLite never intentionally retains references to invalid caches.

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
