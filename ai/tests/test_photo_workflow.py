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
        db, WorkflowFixtureProvider(), aesthetics_service=assessment
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


class WorkflowFixtureProvider(DecisionOpenClipProvider):
    """Explicit protocol stub for model-only wiring; not accuracy evidence."""
    @property
    def model_backed(self):
        return True


def test_pins_draft_and_manual_representatives_are_explicit(workflow):
    db, album, ids, assessment, client = workflow
    current = client.post(f"/workflow/albums/{album}/curations", json={}).json()
    url = f"/workflow/curations/{current['id']}/select"
    available = [p["photo_id"] for p in current["kept"]]
    draft_url = f"/workflow/albums/{album}/selection-draft"
    assert client.get(draft_url).json() is None
    draft = client.post(draft_url, json={"prompt":"选3张", "curation_id":current["id"], "required_photo_ids":[available[-1]]})
    assert draft.status_code == 200
    assert client.get(draft_url).json() == draft.json()
    assert client.post(draft_url, json={"prompt":"选3张", "curation_id":current["id"], "required_photo_ids":["foreign-photo"]}).status_code == 409
    prompt = "挑出最有代表性的3张照片，两个不同的人最少分别有一张"
    assert client.post(url, json={"prompt":prompt}).status_code == 409
    assert client.post(url, json={"prompt":prompt, "representative_photo_ids":[available[0],available[0]]}).status_code == 409
    response = client.post(url, json={"prompt":prompt, "representative_photo_ids":available[:2], "required_photo_ids":[available[-1]]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["feasible"] and len(result["selected"]) == 3
    assert set(result["constraints"]["required_photo_ids"]) == set(available)
    evidence = result["intent_provenance"]["manual_representatives"]
    assert evidence["identity_verified_by_model"] is False
    assert evidence["photo_ids"] == available[:2]
    assert result["intent_provenance"]["original_prompt"] == prompt
    assert client.post(url, json={"prompt":"选2张", "required_photo_ids":["foreign-photo"]}).status_code == 409
    assert client.post(url, json={"prompt":"选1张", "required_photo_ids":available}).status_code == 409


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


def test_proxy_memory_requires_opt_in_and_reports_no_invented_adjustment(workflow):
    _, album, _, _, client = workflow
    curation = client.post(f"/workflow/albums/{album}/curations", json={}).json()
    url = f"/workflow/curations/{curation['id']}/select"
    assert (
        client.post(
            url, json={"prompt": "挑2张城市照片", "allow_proxy_memory": True}
        ).status_code
        == 409
    )
    result = client.post(
        url,
        json={
            "prompt": "挑2张城市照片",
            "allow_proxy_memory": True,
            "use_preference_memory": True,
        },
    )
    assert result.status_code == 200, result.text
    doc = result.json()
    assert doc["intent_provenance"]["preference_profile"] == "assistant-proxy-enabled"
    assert doc["preference_memory"]["allow_proxy"] is True
    assert not doc["preference_memory"]["applied"]


def test_uncertain_categories_do_not_fill_quota_and_old_replacement_is_blocked(workflow):
    from ai.selection.replacement import ReplacementService
    from ai.schemas import SelectionReplacementRequest

    db, album, ids, assessment, client = workflow
    curation = client.post(f"/workflow/albums/{album}/curations", json={}).json()
    url = f"/workflow/curations/{curation['id']}/select"
    # Fixture provider returns identical text vectors: every category is uncertain.
    response = client.post(url, json={"prompt": "选2张，1张人1张风景"})
    assert response.status_code == 200
    assert not response.json()["feasible"] and not response.json()["selected"]
    assert response.json()["constraints"]["category_counts"] == {"portrait": 1, "landscape": 1}
    result = client.post(url, json={"prompt": "选2张城市照片"}).json()
    replacements = ReplacementService(db, WorkflowFixtureProvider(), aesthetics_service=assessment)
    with pytest.raises(ValueError, match="旧版单张替换"):
        replacements.replace(result["selection_id"], SelectionReplacementRequest(remove_photo_id=result["selected"][0]["photo_id"]))


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


@pytest.mark.parametrize("n", [4, 9, 12])
def test_descriptive_count_is_explicit_not_default(n):
    text = f"挑出最适合这趟旅游展示的{n}张照片"
    normalized, intent, defaulted = normalize_query(text, 9, 102)
    assert not defaulted
    assert intent.target_count == n
    assert normalized == f"选{n}张。{text}"


def test_descriptive_count_does_not_hide_conflicting_or_unsupported_conditions():
    from ai.selection.compact import compile_constraints
    from ai.selection.structured import (
        StructuredSelectionDocument,
        _validate_source_coverage,
    )

    normalized, _, _ = normalize_query("选必须全部是建筑的9张照片", 9, 102)
    hard, evidence = compile_constraints(normalized)
    document = StructuredSelectionDocument.model_validate(
        dict(
            hard_constraints=hard,
            constraint_evidence=evidence,
            semantic_query="建筑",
            style_preferences=[],
            unsupported_hard_requirements=[],
            uncertain_requirements=[],
        )
    )
    assert _validate_source_coverage(document, normalized).uncertain_requirements


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
