# Retrieval evaluation end-to-end evidence

Date: 2026-08-14 (Asia/Shanghai)

## Setup

- Album: 81 local JPG/JPEG files derived from the public Wikimedia Commons
  fixture; attribution remains in `.norma/demo-album-eval/ATTRIBUTION.json`.
- Judged set: 72 downloaded images for each of three English queries.
- Labels: binary proxy labels derived from the fixture's download search term.
  These are noisy provenance labels, **not human relevance ground truth**.
- Queries: `travel architecture`, `city night photography`, and
  `mountain travel landscape`.
- Cutoffs: 5, 10, and 20.

The experiment exercises schema v6, query creation, 216 upserts, provider-safe
ranking, metric computation, immutable run persistence, and report retrieval.

## Macro results

| Provider | MRR | P@5 | P@10 | P@20 | R@5 | R@10 | R@20 | nDCG@5 | nDCG@10 | nDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lightweight | 0.833 | 0.667 | 0.667 | 0.450 | 0.143 | 0.281 | 0.384 | 0.689 | 0.682 | 0.522 |
| OpenCLIP multilingual | 1.000 | 0.867 | 0.800 | 0.700 | 0.189 | 0.348 | 0.608 | 0.872 | 0.822 | 0.746 |

OpenCLIP reused all 81 existing vectors in 49 ms before evaluation. Its
`mountain travel landscape` result reached P@10 1.0 and R@20 0.905 under the
proxy labels. Lightweight's weaker architecture first hit produced MRR 0.5 for
that query family, while its other two queries hit a positive label at rank 1.

Run IDs retained in local test databases:

- Lightweight: `41f0d26c0f5e4dec98d0fc577bc22e4d`
- OpenCLIP: `e4a18791668644ee8d33f521c36d31ad`

## Interpretation

The result demonstrates that the evaluation machinery discriminates between
providers and stores auditable evidence. It does not establish general model
quality: query count is three, labels inherit Wikimedia search bias, binary
categories ignore graded relevance, and nine synthetic quality/duplicate
fixtures are unjudged. A publishable comparison needs independently written
queries and blind human 0..3 labels over a deeper pooled candidate set.
