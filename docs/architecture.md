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
- SQLite is enough for album-sized datasets. Embeddings will be cached as NumPy
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

## Reuse decision

The vendored `crates/pianke-core` retains its MIT notice and supplies tested
hash, quality-scoring, and clustering primitives from `pianke-desktop`. Norma
will integrate these primitives instead of recreating the same algorithms.
