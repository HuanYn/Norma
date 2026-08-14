# People Pipeline E2E — 2026-08-14

This smoke test uses 30 Wikimedia Commons JPEGs downloaded with portrait-focused
search terms. All 30 manifest entries contain a source page and license name.
The separate 34-image evaluation fixture adds four exact copies selected from
images where the configured detector finds exactly one face.

## Results

Provider: `opencv-haar-dct-v1` (CPU fallback)

| Metric | Result |
| --- | ---: |
| Evaluation images | 34 |
| Detected faces | 29 |
| Person clusters, including singletons | 25 |
| Multi-face clusters | 4 |
| People indexing duration | 4,172 ms |
| Unexpected multi-face merges | 0 observed |

The four multi-face clusters were exactly the four controlled pairs:

- `004_fd317e72f8ff.jpg` + `eval_person_duplicate_01.jpg`
- `007_76ca7ffa8000.jpg` + `eval_person_duplicate_02.jpg`
- `008_16507d83ac4d.jpg` + `eval_person_duplicate_03.jpg`
- `009_af5255576624.jpg` + `eval_person_duplicate_04.jpg`

This validates exact-copy stability and conservative clustering only. It does
not establish real-world face-recognition accuracy across pose, age, lighting,
or camera changes.

## HTTP verification

A real Uvicorn process indexed the 34-image fixture through `POST /albums/index`
and `POST /albums/{album_id}/people/index`. The API returned 29 faces and 25
clusters in 4,172 ms. A returned crop URL was fetched with HTTP 200,
`image/jpeg`, and 13,274 bytes.

## Integrity gates

- Face source size and modification time are checked before and after detection.
- Provider outputs must match the declared dimension and contain finite values.
- Face crops and descriptors are written below `.norma/`, never beside sources.
- Re-indexing the album removes old face/cluster database records.
- The test treats only controlled exact pairs as known positive identities and
  does not label arbitrary public portraits as the same person.
