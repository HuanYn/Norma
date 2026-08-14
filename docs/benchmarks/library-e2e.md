# Library E2E — 2026-08-14

This smoke test uses openly licensed Wikimedia Commons JPEGs downloaded by
`scripts/download_demo_album.py`. Per-image source, creator and license data is
kept in the untracked fixture's `ATTRIBUTION.json`.

## Datasets

| Fixture | Images | Purpose |
| --- | ---: | --- |
| `demo-album` | 72 | Natural public-photo indexing |
| `demo-album-eval` | 81 | Natural photos plus 4 exact duplicates and 5 controlled quality derivatives |

The evaluation fixture is a separate copy. Downloaded source JPEGs are not
modified. Licenses represented in the 72-image source fixture include CC BY,
CC BY-SA, CC0, Public Domain, and no-known-restrictions records.

## Results

Provider: `pillow-opencv-fallback-v1` (CPU)

| Metric | Natural album | Controlled evaluation album |
| --- | ---: | ---: |
| Indexed JPEGs | 72 | 81 |
| Duration | 6,085 ms | 6,805 ms |
| Errors | 0 | 0 |
| Suggested exclusions | 0 | 4 |
| Similarity groups | 0 | 7 |
| Photos assigned to a similarity group | 0 | 14 |

The natural set intentionally contains unrelated, generally high-quality
Commons images, so zero exclusions and zero duplicate groups are credible.
Controlled derivatives supply deterministic positive cases for the UI and
regression gates; they are not presented as natural benchmark discoveries.

## HTTP verification

The real FastAPI process indexed the 81-image fixture through
`POST /albums/index` in 6,708 ms. It returned schema version 2, four suggested
exclusions, seven similarity groups, and no errors. A returned thumbnail URL
was fetched from `/media/...` with HTTP 200, `image/jpeg`, valid JPEG magic, and
53,974 bytes.

## Integrity gates

- 72/72 source files decode as JPEG.
- 72/72 manifest entries include a source page and license name.
- Source file size and modification time are checked before and after indexing.
- Thumbnails and SQLite data are written only below `.norma/`.
- Original deletion is not implemented anywhere in the Library pipeline.
