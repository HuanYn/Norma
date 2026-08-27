# Incremental people pipeline end-to-end evidence

Date: 2026-08-14 (Asia/Shanghai)

## Setup

- Album: 34 local JPG/JPEG files copied from the public Wikimedia portrait
  fixture. Attribution remains with the source fixture under `.norma/`.
- Provider: `opencv-haar-dct-v1`, CPU, conservative Haar detection and
  79-dimensional descriptors.
- State: fresh schema v7 database under `.norma/people-incremental-e2e`.

## Results

| Run | Index | People | Computed photos | Reused photos | Faces | Clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold | 4,121 ms | 4,667 ms | 34 | 0 | 29 | 25 |
| Unchanged | — | 79 ms | 0 | 34 | 29 | 25 |
| One source replaced + re-indexed | 228 ms | 468 ms | 1 | 33 | 29 | 25 |

The unchanged people stage was about 98.3% faster than cold processing. After
one disposable fixture photo was replaced, indexing and face detection each
computed exactly one photo and reused 33. Global clustering still rebuilt over
all 29 descriptors, so the fast path does not preserve stale membership.

## Failure and integrity checks

- A cached zero-face result is reusable because `face_processed` distinguishes
  "not run" from "ran and found zero".
- Deleting one descriptor recomputes only its owning photo.
- Provider mismatch, recorded face-count mismatch, missing crop, invalid vector,
  or source fingerprint mismatch invalidates that photo.
- Modifying a source after indexing blocks people analysis until re-index.
- Cancellation is honored between photos and before the final transaction.
- Re-indexing a changed photo deletes only that photo's faces and clears cluster
  assignments; stable descriptors remain available for the next cluster rebuild.

This provider remains plumbing-level face grouping rather than biometric
identity. Counts and speed validate lifecycle behavior, not identity accuracy.
