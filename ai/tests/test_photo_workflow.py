"""Controlled workflow contracts; test embeddings/scores are not accuracy evidence."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from ai.config import Settings
from ai.selection.service import SelectionService
from ai.tests.test_learned_selection import setup
from ai.tests.test_contextual_decision_runtime import DecisionOpenClipProvider
from ai.workflow import CurationRequest, curate, normalize_query, create_workflow_router


@pytest.fixture
def workflow(tmp_path):
    db, album, ids, assessment = setup(tmp_path)
    selection = SelectionService(
        db, DecisionOpenClipProvider(), aesthetics_service=assessment
    )
    app = FastAPI()
    app.include_router(
        create_workflow_router(
            lambda: db,
            lambda: assessment,
            lambda: selection,
            lambda: Settings(data_dir=tmp_path),
        )
    )
    return db, album, ids, assessment, TestClient(app)


def test_best_per_group_filters_view_without_deleting_photos(workflow):
    db, album, ids, assessment, client = workflow
    with db.connect() as c:
        c.execute(
            "UPDATE photos SET similarity_group='near-duplicates' WHERE id IN (?,?)",
            (ids["a"], ids["b"]),
        )
    before = {
        r["absolute_path"]: Path(r["absolute_path"]).read_bytes()
        for r in assessment._rows(album)
    }
    result = client.post(f"/workflow/albums/{album}/curations", json={}).json()
    assert (
        result["total"] == 4 and len(result["kept"]) == 3 and result["group_count"] == 1
    )
    assert ids["b"] in [p["photo_id"] for p in result["kept"]]
    assert ids["a"] in [p["photo_id"] for p in result["hidden"]]
    assert result["originals_deleted"] == 0
    assert all(Path(p).read_bytes() == content for p, content in before.items())


def test_selection_calls_provider_and_is_confined_to_curated_ids(workflow):
    db, album, ids, assessment, client = workflow
    curation = client.post(f"/workflow/albums/{album}/curations", json={}).json()
    response = client.post(
        f"/workflow/curations/{curation['id']}/select", json={"prompt": "挑2长城市照片"}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["constraints"]["target_count"] == 2
    assert result["intent_provenance"]["effective_prompt"] == "挑2张城市照片"
    assert set(p["photo_id"] for p in result["selected"]) <= set(
        p["photo_id"] for p in curation["kept"]
    )
    assert result["learned_quality"] and result["provider_fingerprint"].startswith(
        "openclip"
    )
    assert (
        client.get(
            f"/workflow/selections/{result['selection_id']}/images/{result['selected'][0]['photo_id']}"
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/workflow/selections/{result['selection_id']}/images/{ids['a']}"
        ).status_code
        == 404
    )


def test_default_group_count_and_excess_count_are_visible(workflow):
    _, album, _, _, client = workflow
    curation = client.post(f"/workflow/albums/{album}/curations", json={}).json()
    url = f"/workflow/curations/{curation['id']}/select"
    r = client.post(url, json={"prompt": "挑一组适合发旅游朋友圈的照片"})
    assert r.status_code == 200, r.text
    assert r.json()["constraints"]["target_count"] == 3
    assert r.json()["intent_provenance"]["default_count_applied"]
    assert client.post(url, json={"prompt": "挑4张雪山照片"}).status_code == 409


def test_stale_curation_and_image_are_rejected(workflow):
    db, album, ids, _, client = workflow
    curation = client.post(f"/workflow/albums/{album}/curations", json={}).json()
    with db.connect() as c:
        c.execute("UPDATE photos SET auto_reject=1 WHERE id=?", (ids["b"],))
    assert client.get(f"/workflow/curations/{curation['id']}").status_code == 409
    assert (
        client.post(
            f"/workflow/curations/{curation['id']}/select",
            json={"prompt": "挑2张旅行照"},
        ).status_code
        == 409
    )


def test_unassessed_and_invalid_thresholds_fail(workflow):
    db, album, ids, _, client = workflow
    assert (
        client.post(
            f"/workflow/albums/{album}/curations", json={"min_aesthetic": 11}
        ).status_code
        == 422
    )
    with db.connect() as c:
        c.execute("DELETE FROM aesthetics_scores_v1 WHERE photo_id=?", (ids["a"],))
    assert (
        client.post(f"/workflow/albums/{album}/curations", json={}).status_code == 409
    )


@pytest.mark.parametrize(
    "text,n",
    [("挑4长雪山照片", 4), ("选出4张雪山", 4), ("挑一组适合发旅游朋友圈的照片", 9)],
)
def test_normalization(text, n):
    normalized, intent, defaulted = normalize_query(text, 9, 125)
    assert intent.target_count == n
    assert ("选9张。" in normalized) == defaulted


def test_125_photos_are_not_truncated(workflow):
    db, album, ids, assessment, _ = workflow
    # Expand DB/file fixtures, not a claim of 125 real pretrained inferences.
    with db.connect() as c:
        original = dict(
            c.execute("SELECT * FROM photos WHERE id=?", (ids["b"],)).fetchone()
        )
        for i in range(121):
            row = original | {
                "id": f"bulk-{i:03d}",
                "absolute_path": original["absolute_path"] + f".{i}",
                "similarity_group": "bulk-duplicates",
            }
            Path(row["absolute_path"]).write_bytes(
                Path(original["absolute_path"]).read_bytes()
            )
            import os

            os.utime(
                row["absolute_path"],
                ns=(row["source_mtime_ns"], row["source_mtime_ns"]),
            )
            columns = ",".join(row)
            c.execute(
                f"INSERT INTO photos({columns}) VALUES ({','.join('?' for _ in row)})",
                list(row.values()),
            )
            score = c.execute(
                "SELECT * FROM aesthetics_scores_v1 WHERE photo_id=?", (ids["b"],)
            ).fetchone()
            values = dict(score) | {"photo_id": row["id"]}
            c.execute(
                f"INSERT INTO aesthetics_scores_v1({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
                list(values.values()),
            )
    result = curate(db, assessment, album, CurationRequest())
    assert result["total"] == 125
    assert len(result["kept"]) + len(result["hidden"]) == 125
    assert sum(p["group"] == "bulk-duplicates" for p in result["kept"]) == 1
