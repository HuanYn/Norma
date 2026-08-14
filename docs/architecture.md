# Norma Architecture

Norma is a local-first desktop application. The first portfolio milestone keeps
the system deliberately small: one Vue/Tauri shell, one local Python process,
and one SQLite database.

## Process boundary

```text
Vue UI
  │ Tauri invoke
  ▼
Rust desktop shell
  │ loopback HTTP (127.0.0.1 only)
  ▼
FastAPI worker
  │
  ├── SQLite metadata and feedback
  └── cache/thumbnails/embeddings (never the source folder)
```

The Rust shell owns the worker lifecycle. The UI never imports a model or API
client. Future closed and local models sit behind Python provider interfaces.

## Data ownership

- Source JPG/JPEG files are read-only inputs.
- Generated thumbnails, embeddings, and the database live under `.norma/` by
  default and are ignored by Git.
- SQLite is enough for album-sized datasets. Embeddings are cached as NumPy
  arrays and searched with cosine similarity before adding any vector database.

## Milestone boundaries

1. **M0:** desktop shell, worker health, SQLite schema, verified bridge.
2. **M1:** JPG scan, thumbnails, quality signals, similarity groups, reject fold.
3. **M2:** multimodal embeddings, people clusters, text/image retrieval.
4. **M3:** structured constraints, scoring, CP-SAT collection optimization.
5. **M4:** grounded explanations, replacement, pairwise preference learning.

RL, video, and world generation remain deferred until the personalized photo
selection loop is demonstrably complete.

## Library fast path and fallback

`crates/pianke-core` is the intended production fast path for hashes, quality
scoring, and clustering. Until the Windows SDK linker is available on a new
development machine, `ai/index/` provides an explicit
`pillow-opencv-fallback-v1` implementation. Both write the same photo-domain
fields so transport and UI code do not depend on the provider. The fallback:

- recursively discovers JPG/JPEG files without a hard album-size limit;
- verifies source size and modification time before and after every read;
- writes thumbnails only under `.norma/data/thumbnails/`;
- computes conservative quality suggestions and never deletes a source;
- assigns similarity groups from pHash and dHash distance.

The public demo downloader stores image-level creator, source and license
metadata in `.norma/demo-album/ATTRIBUTION.json`. The dataset is a development
fixture, not a redistribution bundle committed to this repository.

## Retrieval provider boundary

`ai/index/embedding.py` defines a shared image/text embedding interface. The
default `lightweight-semantic-v1` implementation produces deterministic,
normalized 16-dimensional CPU descriptors for color, luminance, composition,
and coarse visual concepts. It is an interpretable integration baseline rather
than a claim of CLIP-level open-vocabulary understanding.

`ai/retrieval/` owns cache generation and exact cosine search. Cache paths are
provider-scoped below `.norma/data/embeddings/`; the API supports text queries,
reference-photo queries, and an optional candidate subset. This boundary lets a
future CLIP/SigLIP provider replace descriptor generation without changing the
desktop transport or result schema.

## People provider boundary

`ai/people/` separates detection/description from persistence and clustering.
The default `opencv-haar-dct-v1` CPU fallback detects frontal faces and stores a
79-dimensional DCT/color descriptor. Exact cosine clustering uses a conservative
`0.985` threshold; single-face clusters remain visible rather than being forced
into a larger identity group. This is pipeline scaffolding, not biometric
identification. A future face-embedding provider can replace it behind the same
interface and SQLite `faces` / `person_clusters` tables.

Face descriptors and crops are provider-scoped under `.norma/data/faces/`.
Re-indexing an album invalidates both semantic embeddings and people records;
generated cache files are disposable, while the database never continues to
reference stale analysis.

## Structured selection boundary

`ai/selection/parser.py` turns a bounded bilingual prompt grammar into explicit
hard constraints: exact target count, quality floor, reject policy, and maximum
photos per similarity group. Unrecognized requested concepts fail visibly;
only a prompt containing no semantic request may fall back to quality-only
ranking.

`ai/selection/service.py` combines semantic cosine similarity and normalized
quality into an auditable soft score. `optimizer.py` then enforces hard
constraints over the complete collection. With the optional `selection`
dependency it uses OR-Tools CP-SAT with integer objective coefficients,
single-threaded deterministic settings, and a two-second limit. Otherwise a
deterministic partition-matroid greedy solver is exact for the current
cardinality plus per-group-capacity constraint family. Infeasible requests
return no partial selection. Parsed constraints and full results are persisted
in SQLite `selections`.

## Preference and replacement loop

`ai/preferences/` represents each photo with seven bounded, inspectable
features: semantic relevance, quality, sharpness, brightness, contrast, and
landscape/portrait orientation. Pairwise feedback applies an online logistic
update with learning-rate decay, light regularization, and clipped weights.
The local model and raw feature difference are persisted in
`user_preferences` and `feedback`; no face descriptor or preference event is
sent off-device.

Once comparisons exist, selection reserves 13% of its soft score for the local
preference probability while preserving all M3 hard constraints. Explanations
report semantic similarity, quality, constraint eligibility, preference fit,
and the number of comparisons supporting that fit.

`ai/selection/replacement.py` treats every unremoved selected photo as locked.
It chooses the highest-scoring unselected candidate that preserves the original
reject policy, quality floor, and similarity-group capacity. A successful
replacement becomes a new immutable selection audit. If no candidate is
eligible, the endpoint returns infeasible with no partial or silently changed
collection.

## Reuse decision

The vendored `crates/pianke-core` retains its MIT notice and supplies tested
hash, quality-scoring, and clustering primitives from `pianke-desktop`. Norma
will integrate these primitives instead of recreating the same algorithms.
