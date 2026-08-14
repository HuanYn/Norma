# Incremental prepare and recovery benchmark

This benchmark uses the ignored 81-image Wikimedia Commons evaluation album.
It measures the same read-only folder prepared twice in separate CLI processes.
The second run has no source changes.

## Lightweight provider

| Stage | First run | Unchanged run | Reused |
| --- | ---: | ---: | ---: |
| index | 9,845 ms | 133 ms | 81/81 |
| embedding | 6,960 ms | 71 ms | 81/81 |
| combined core stages | 16,805 ms | 204 ms | 81/81 |

The unchanged core pipeline was about 98.8% faster. A text search immediately
after the reused run succeeded, confirming that cache reuse did not merely skip
work while leaving the album unavailable.

## Multilingual OpenCLIP provider

The model was already downloaded, `HF_HUB_OFFLINE=1` was enabled, and both runs
used CPU with 8-image model batches.

| Stage | First run | Unchanged run | Reused |
| --- | ---: | ---: | ---: |
| index | 9,809 ms | 129 ms | 81/81 |
| embedding | 57,788 ms | 88 ms | 81/81 |
| combined core stages | 67,597 ms | 217 ms | 81/81 |

Because all 81 vectors were current, the second process validated the cached
vectors without loading OpenCLIP into memory. The core stages were about 99.7%
faster than the first run.

## Recovery and cancellation evidence

Deterministic tests use a provider with one photo per chunk:

- an injected failure in chunk 2 leaves chunk 1 committed; retry reports
  `reused_count=1` and `computed_count=2` for the three-photo album;
- a cancellation requested after chunk 1 stops before chunk 2 and preserves the
  one completed vector;
- changing one of two indexed photos causes exactly one vector to be recomputed;
- changing a source without re-indexing blocks both search and embedding rather
  than blessing a stale fingerprint.

## Safety boundary

SQLite schema v5 records the indexed source size and nanosecond modification
time, plus the source fingerprint used for each embedding. Search, selection,
replacement, and preference feedback require all values to match the current
source and active provider. Cache files are written under unique names and a
chunk's SQLite references change only after every vector in that chunk is
validated and saved.

The fingerprint is intentionally inexpensive, not cryptographic. An external
tool that changes image bytes while deliberately restoring both exact size and
nanosecond modification time can evade it. Normal filesystem edits, copies, and
exports change at least one value and are detected.
