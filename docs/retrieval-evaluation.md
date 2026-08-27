# Retrieval relevance evaluation

Norma stores relevance evidence locally so retrieval changes can be compared
with repeatable metrics instead of screenshots or hand-picked examples.

## Workflow

1. Prepare an album with the provider being evaluated.
2. Create realistic queries before inspecting the ranking.
3. Request up to 50 candidates for each query.
4. Assign grades from 0 to 3. Include known relevant misses outside the first
   page when possible.
5. Run a report at fixed cutoffs and retain its run ID.

The website can call the same workflow through:

```text
POST /evaluation/queries
GET  /albums/{album_id}/evaluation/queries
GET  /evaluation/queries/{query_id}/candidates?limit=50
PUT  /evaluation/queries/{query_id}/judgments
POST /albums/{album_id}/evaluation/runs
GET  /evaluation/runs/{run_id}
```

A judgment batch is idempotent by `(query_id, photo_id)`: submitting a photo
again updates its grade and annotator. Photos from another album are rejected.
Duplicate normalized query text within an album is rejected to avoid accidental
double weighting.

## Metrics

- `Precision@K`: fraction of the first K ranks with grade greater than zero.
- `Recall@K`: relevant judged photos recovered by K divided by all positively
  judged photos for that query.
- `nDCG@K`: graded ranking quality with gain `2^grade - 1` and logarithmic rank
  discount.
- `MRR`: reciprocal rank of the first positively judged result.

Macro fields are unweighted means across queries with at least one judgment.
A judged query with no positive grade contributes zero. A query with no labels
is skipped and reported in `skipped_query_count`.

## Integrity boundaries

- Candidate generation and report runs require the active embedding provider
  and a complete source-fingerprint-current cache.
- Unjudged ranked photos are treated as grade zero. This is conventional for a
  fixed judgment pool but can underestimate recall on shallow pools.
- A persisted run contains provider identity, ranked photo IDs, relevance
  snapshot, cutoffs, and metrics. It is an audit record, not a claim that labels
  are correct.
- Download search categories, filenames, or metadata may be useful proxy labels
  for pipeline testing, but must not be described as human ground truth.
