# Retrieval E2E — 2026-08-14

This evaluation uses the 81-image public fixture described in
`library-e2e.md`: 72 openly licensed Wikimedia Commons JPEGs, four exact copies,
and five controlled quality derivatives. JPEGs and attribution manifests remain
untracked under `.norma/`.

## Configuration

| Item | Value |
| --- | --- |
| Provider | `lightweight-semantic-v1` |
| Runtime | CPU, deterministic |
| Vector dimension | 16 |
| Similarity | exact cosine / normalized dot product |
| Indexed images | 81 |
| Embedding duration | 4,930 ms |
| Embedding errors | 0 |

The provider is a lightweight integration baseline, not a pretrained
vision-language model. Results measure whether the cache, API, ranking, and UI
pipeline work end to end and expose the baseline's qualitative limits.

## Text retrieval observations

Top-five checks against source titles:

| Query | Relevant observations | Known errors |
| --- | --- | --- |
| `夜景 dark night` | ranks both controlled dark images first; the next three are titled Night Photography City, Night time in a city, and New York City at night | darkness can rank non-night underexposed photos |
| `建筑 architecture city` | retrieves a controlled copy/source pair, Radio City Music Hall, and Kewick Long Building | a mountain artwork appears fourth |
| `自然 green mountain` | top five includes three explicitly titled mountain/landscape images | a blurred derivative and a transit map rank above them |
| `电影感 cinematic warm` | retrieves warm, high-contrast night/landscape material | cinematic intent remains only a visual heuristic |

These are directional smoke checks, not precision/recall claims. The controlled
and source filenames were joined to `ATTRIBUTION.json` only for evaluation; the
search ranker does not read filenames, titles, search terms, or attribution.

## Image retrieval check

Using `001_302770d7dacd.jpg` as the reference, its controlled exact copy
`eval_duplicate_01.jpg` ranks first with cosine similarity `1.000000`. The
reference itself is excluded from results. A separate API regression test also
verifies that the reference may sit outside an explicitly supplied candidate
subset.

## Regression gates

- Text and image modes share the same provider-scoped cache.
- Every loaded vector must have dimension 16 and contain only finite values.
- Search before embedding returns a clear HTTP 404.
- Supplying both text and a reference photo returns HTTP 400.
- A query with no recognized baseline concept returns HTTP 400 instead of a
  fabricated hash-based ranking.
- Original file size and modification time are checked before and after image
  descriptor extraction.
