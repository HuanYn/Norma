# Maintenance audit and quota evidence

Date: 2026-08-26 (Asia/Shanghai)

## State snapshot

The real 81-image lightweight evaluation database was inspected while pointing
at the shared offline OpenCLIP model directory.

| Category | Files | Bytes |
| --- | ---: | ---: |
| Thumbnails | 81 | 2,868,471 |
| Embeddings | 81 | 15,552 |
| Faces | 0 | 0 |
| Models | 23 | 2,957,610,782 |
| SQLite | — | 331,776 |
| Total | 185 + SQLite | 2,960,826,581 |

## Conservative 1 GiB policy

- Budget: 1,073,741,824 bytes.
- Over budget: 1,887,084,757 bytes.
- Generated scan: 162 files, all 162 referenced.
- Eligible orphans: 0 files / 0 bytes.
- Projected total after safe GC: unchanged at 2,960,826,581 bytes.
- Result: `satisfied=false`, `projected_satisfied=false`.

The service correctly explained that orphan cleanup cannot meet the budget
because the model and referenced state dominate usage. It did not delete any
model or live cache file.

The dry-run was persisted as one completed `quota_enforce` maintenance record
with its exact request, full result snapshot, timestamps, and no error. This
demonstrates audit persistence rather than only an ephemeral CLI printout.
