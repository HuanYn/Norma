# Library lifecycle public-data E2E

Date: 2026-08-14. The existing 81-image Wikimedia Commons evaluation fixture
was submitted to `POST /jobs/prepare` with people grouping disabled. Runtime
state used an isolated `.norma/lifecycle-e2e` directory.

## Result

| Check | Observed |
|---|---:|
| Submit response | HTTP 202 |
| Indexed photos | 81 |
| Suggested rejects | 4 |
| Cached embeddings | 81 |
| Index duration | 10,306 ms |
| Embedding duration | 5,488 ms |
| Final state | `completed`, progress `1.0` |
| Quality-sorted first page | 10 of 81 |

Observed persisted stages were `running/indexing` at 0.05,
`running/embedding` at 0.55, and `completed` at 1.0. The people stage was
intentionally skipped for this run.

The FastAPI lifespan was then closed and started again against the same SQLite
database. After restart:

- `GET /jobs/{id}` still returned the completed job and compact result;
- `GET /albums/{id}` returned 81 photos, 4 rejects, and 81 embeddings;
- `GET /jobs?status=completed` returned the persisted job;
- paginated quality sorting returned 10 items and a total of 81.

A separate real Uvicorn TCP smoke test then served the same state on port 8881.
`GET /albums?limit=1`, a three-item photo page, and
`GET /jobs?status=completed` all returned successfully with totals 1, 81, and 1.

The known Pillow `Truncated File Read` EXIF warning occurred for one decodable
public JPEG, matching earlier fixture runs. It did not create an indexing error
or alter the source image.
