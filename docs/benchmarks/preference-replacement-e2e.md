# Preference + Replacement E2E — 2026-08-14

This M4 smoke test uses the same public 81-image album and a 12-photo night
selection. Among the selected set, the highest-quality photo was preferred over
the lowest-quality photo three times to provide a controlled directional signal.

## Pairwise result

| Item | Value |
| --- | --- |
| Preferred | `024_3fd5f118eb04.jpg`, quality 80.918 |
| Rejected | `033_1bcb794cd740.jpg`, quality 48.635 |
| Comparisons | 3 |
| Positive learned weights | quality, sharpness, brightness, contrast |
| Negative learned weight | semantic relevance |

The negative semantic weight is expected evidence, not an error: in this pair,
the user explicitly preferred the clearer but slightly less semantically similar
photo. After three comparisons, the selected membership still overlapped the
baseline 12/12; the system therefore reports score personalization without
claiming that this small signal changed the set.

Top personalized results include grounded reasons such as:

- `semantic similarity 0.365`
- `quality 75.4/100`
- `passes reject and collection constraints`
- `personal preference fit 0.589 from 3 comparisons`

## Replacement result

Removing `024_3fd5f118eb04.jpg` produced
`041_00aa49cb3758.jpg` as the replacement. All 11 non-removed photos were
preserved. The replacement explanation includes semantic similarity 0.264,
quality 69.8, constraint eligibility, and preference fit 0.579 from three
comparisons.

## HTTP verification

A real Uvicorn process completed a six-photo CP-SAT selection, accepted one
pairwise feedback event, and returned a feasible locked-set replacement. The
updated selection still contained six photos and returned six grounded
explanation entries. The feedback event had a persistent ID and advanced the
local model comparison count.

## Gates

- Preferred and rejected IDs must be different and belong to the album.
- A supplied selection context must belong to the same album.
- Preference features and weight updates are persisted locally and remain
  inspectable.
- Replacement never changes locked photos.
- No eligible replacement returns infeasible and no partial selection.
- A successful replacement creates a new selection audit instead of mutating
  prior history.
