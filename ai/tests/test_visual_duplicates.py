"""Synthetic contract checks; not a real-photo duplicate accuracy benchmark."""

import numpy as np
from PIL import Image
import pytest

from ai.selection.visual_duplicates import features, spatial_evidence, fold_similar
from ai.tests.test_photo_workflow import WorkflowFixtureProvider
from ai.tests.test_learned_selection import setup
from ai.workflow import curate, CurationRequest


def test_spatial_verification_allows_crop_but_not_tiny_shared_region(tmp_path):
    rng = np.random.default_rng(12)
    pixels = rng.integers(0, 256, (400, 400), dtype=np.uint8)
    whole, crop, tiny, different = [
        tmp_path / f"{name}.png" for name in ["whole", "crop", "tiny", "different"]
    ]
    Image.fromarray(pixels).save(whole)
    Image.fromarray(pixels[40:360, 40:360]).save(crop)
    Image.fromarray(pixels[140:260, 140:260]).save(tiny)
    Image.fromarray(rng.integers(0, 256, (400, 400), dtype=np.uint8)).save(different)
    assert spatial_evidence(features(whole), features(crop))["matched"]
    for other in [tiny, different]:
        evidence = spatial_evidence(features(whole), features(other))
        assert evidence is None or not evidence["matched"]


def test_quality_first_non_transitive_folding_and_vector_snapshot(tmp_path):
    db, album, ids, assessment = setup(tmp_path)
    with db.connect() as c:
        rows = c.execute(
            "SELECT * FROM photos WHERE album_id=? ORDER BY id", (album,)
        ).fetchall()
    baseline = curate(db, assessment, album, CurationRequest())
    kept = baseline["kept"]
    by_id = {r["id"]: r for r in rows}
    for index, p in enumerate(kept):
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = np.cos(index * 0.2)
        vector[1] = np.sin(index * 0.2)
        np.save(by_id[p["photo_id"]]["embedding_path"], vector)
    retained, folded, evidence = fold_similar(kept, rows, WorkflowFixtureProvider())
    assert len(retained) == 2 and len(folded) == 1
    assert folded[0]["visual_duplicate_of"] == kept[0]["photo_id"]
    assert kept[2]["photo_id"] in [p["photo_id"] for p in retained]
    assert evidence["transitive_merging"] is False
    # Only vector bytes changed: curation identity must change too.
    before = curate(db, assessment, album, CurationRequest(), WorkflowFixtureProvider())
    vector = np.zeros(512, dtype=np.float32)
    vector[2] = 1
    np.save(by_id[kept[1]["photo_id"]]["embedding_path"], vector)
    after = curate(db, assessment, album, CurationRequest(), WorkflowFixtureProvider())
    assert before["snapshot_sha256"] != after["snapshot_sha256"]


def test_visual_curation_requires_current_learned_cache(tmp_path):
    db, album, ids, assessment = setup(tmp_path)
    with db.connect() as c:
        c.execute(
            "UPDATE photos SET embedding_provider=? WHERE id=?",
            ("wrong-provider", ids["b"]),
        )
    with pytest.raises(ValueError, match="图片特征"):
        curate(db, assessment, album, CurationRequest(), WorkflowFixtureProvider())
