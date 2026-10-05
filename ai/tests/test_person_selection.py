from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai import app as app_module
from ai.people.indexer import PeopleIndexer
from ai.people.labels import PersonLabelService
from ai.people.provider import DetectedFace
from ai.preferences.suggestion_service import (
    PreferenceSuggestionConflictError,
    PreferenceSuggestionService,
)
from ai.schemas import (
    SelectionRequest,
    SelectionReplacementRequest,
    PreferencePairSuggestionRequest,
)
from ai.selection.optimizer import OptimizationCandidate, optimize_collection
from ai.selection.parser import parse_selection_prompt, has_semantic_content
from ai.selection.replacement import ReplacementService
from ai.selection.service import SelectionService
from ai.tests.test_people import FakeFaceProvider
from ai.tests.test_selection import _album, FakeSelectionProvider


class QuotaFaceProvider(FakeFaceProvider):
    def detect(self, path: Path) -> list[DetectedFace]:
        vector = [1.0, 0.0, 0.0] if path.stem in {"c", "e", "f"} else [0.0, 1.0, 0.0]
        return [
            DetectedFace(
                box=(10, 10, 40, 40),
                descriptor=np.asarray(vector, dtype=np.float32),
                crop=Image.new("RGB", (40, 40), "tan"),
            )
        ]


def _named_album(tmp_path: Path):
    database, album_id, ids = _album(tmp_path)
    indexer = PeopleIndexer(database, tmp_path / "data", QuotaFaceProvider())
    people = indexer.index(album_id)
    cluster = next(
        c for c in people.clusters if any(f.photo_id == ids["c"] for f in c.faces)
    )
    PersonLabelService(database).set(album_id, cluster.cluster_id, "Me")
    return database, album_id, ids, indexer, cluster.cluster_id


@pytest.mark.parametrize(
    "prompt,key",
    [
        ("选3张，至少2张有我", "me"),
        ("选3张，至少2张有“Alex”", "alex"),
        ('pick 3 photos, at least 2 photos with "Alex"', "alex"),
        ("pick 3 photos, at least 2 photos with me", "me"),
    ],
)
def test_person_prompt_is_hard_not_semantic(prompt, key):
    intent = parse_selection_prompt(prompt)
    assert intent.person_minimums == {key: 2}
    assert not has_semantic_content(prompt)


def test_person_quota_before_total_does_not_change_total_count():
    result = parse_selection_prompt("at least 2 photos with me, select 5 photos")
    assert result.target_count == 5
    assert result.person_minimums == {"me": 2}


@pytest.mark.parametrize(
    "prompt",
    [
        "选9张，至少2张夜景",
        "选9张，不要自拍",
        "选9张，至少2张有Alex",
        "pick 9 photos, at least 2 photos of night",
        "pick 9 photos, no selfies",
        "选9张，最多2张有我",
        "选9张，至少两张有我",
        "选9张，不要有自拍",
    ],
)
def test_unimplemented_hard_conditions_are_not_silently_accepted(prompt):
    with pytest.raises(ValueError, match="Unsupported hard constraint"):
        parse_selection_prompt(prompt)


def test_person_quota_changes_set_and_survives_refresh_and_replacement(tmp_path):
    database, album_id, ids, indexer, cluster_id = _named_album(tmp_path)
    service = SelectionService(database, FakeSelectionProvider())
    baseline = service.select(SelectionRequest(album_id=album_id, prompt="选2张"))
    assert (
        len({ids["c"], ids["e"], ids["f"]} & {p.photo_id for p in baseline.selected})
        == 1
    )
    result = service.select(
        SelectionRequest(album_id=album_id, prompt="选2张，至少2张有我")
    )
    assert result.feasible
    assert all(p.photo_id in {ids["c"], ids["e"], ids["f"]} for p in result.selected)
    assert result.people_snapshot_sha256
    assert service.get(result.selection_id) == result
    assert (
        next(
            c for c in indexer.index(album_id).clusters if c.cluster_id == cluster_id
        ).label
        == "Me"
    )
    replace = ReplacementService(database, FakeSelectionProvider()).replace(
        result.selection_id, SelectionReplacementRequest(remove_photo_id=ids["c"])
    )
    assert replace.feasible
    assert replace.replacement.photo_id == ids["e"]
    assert replace.updated_selection.constraints.person_minimums == {"me": 2}
    assert all(
        p.photo_id in {ids["c"], ids["e"], ids["f"]}
        for p in replace.updated_selection.selected
    )


def test_quota_and_quality_conflict_returns_no_partial_set(tmp_path):
    database, album_id, _, _, _ = _named_album(tmp_path)
    result = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(album_id=album_id, prompt="选3张，质量至少50，至少3张有我")
    )
    assert not result.feasible
    assert result.selected == []


def test_quota_respects_subset_and_missing_identity(tmp_path):
    database, album_id, ids, _, cluster_id = _named_album(tmp_path)
    service = SelectionService(database, FakeSelectionProvider())
    result = service.select(
        SelectionRequest(
            album_id=album_id,
            prompt="选2张",
            person_minimums={"Me": 2},
            subset_photo_ids=[ids["a"], ids["b"], ids["c"]],
        )
    )
    assert not result.feasible
    PersonLabelService(database).set(album_id, cluster_id, "Unknown")
    with pytest.raises(ValueError, match="尚未命名"):
        service.select(SelectionRequest(album_id=album_id, prompt="选2张，至少1张有我"))


def test_unanalysed_people_is_not_absence(tmp_path):
    database, album_id, _ = _album(tmp_path)
    with pytest.raises(ValueError, match="完整的人脸分析"):
        SelectionService(database, FakeSelectionProvider()).select(
            SelectionRequest(album_id=album_id, prompt="选2张，至少1张有我")
        )


def test_replacement_rejects_renamed_evidence(tmp_path):
    database, album_id, _, _, cluster_id = _named_album(tmp_path)
    result = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(album_id=album_id, prompt="选2张，至少2张有我")
    )
    labels = PersonLabelService(database)
    labels.set(album_id, cluster_id, "Alex")
    labels.set(album_id, cluster_id, "Me")
    with pytest.raises(ValueError, match="证据已变化"):
        ReplacementService(database, FakeSelectionProvider()).replace(
            result.selection_id,
            SelectionReplacementRequest(remove_photo_id=result.selected[0].photo_id),
        )


def test_named_source_content_change_is_rejected(tmp_path):
    database, album_id, _, _, _ = _named_album(tmp_path)
    import os

    photo = tmp_path / "album" / "c.jpg"
    stat = photo.stat()
    content = bytearray(photo.read_bytes())
    content[-20] ^= 1
    photo.write_bytes(content)
    os.utime(photo, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="source evidence changed"):
        SelectionService(database, FakeSelectionProvider()).select(
            SelectionRequest(album_id=album_id, prompt="选2张，至少1张有我")
        )


def test_overlapping_person_quotas_count_distinct_photos():
    candidates = [
        OptimizationCandidate(0, 1, "a", frozenset({"me", "alex"})),
        OptimizationCandidate(1, 0.8, "b", frozenset({"me", "alex"})),
        OptimizationCandidate(2, 2, "c", frozenset()),
    ]
    result = optimize_collection(candidates, 2, 1, {"me": 2, "alex": 2})
    assert set(result.indices) == {0, 1}
    grouped = [
        OptimizationCandidate(c.index, c.score, "same", c.person_labels)
        for c in candidates
    ]
    assert optimize_collection(grouped, 2, 1, {"me": 2}).status == "infeasible"


def test_legacy_acquisition_cannot_ignore_person_constraints(tmp_path):
    database, album_id, _, _, _ = _named_album(tmp_path)
    selected = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(album_id=album_id, prompt="选2张，至少2张有我")
    )
    with pytest.raises(
        PreferenceSuggestionConflictError, match="does not support person quotas"
    ):
        PreferenceSuggestionService(
            database, FakeSelectionProvider(), preference_mode="adaptive"
        ).suggest(selected.selection_id, PreferencePairSuggestionRequest())


def test_person_label_http_roundtrip_and_revision(tmp_path, monkeypatch):
    database, album_id, _, indexer, cluster_id = _named_album(tmp_path)
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "people_indexer", lambda: indexer)
    with TestClient(app_module.app) as client:
        response = client.patch(
            f"/albums/{album_id}/people/{cluster_id}",
            json={"label": "Alex", "expected_revision": 1},
        )
        assert response.status_code == 200
        assert response.json()["label_status"] == "confirmed"
        assert response.json()["label_revision"] == 2
        stale = client.patch(
            f"/albums/{album_id}/people/{cluster_id}",
            json={"label": "Me", "expected_revision": 1},
        )
        assert stale.status_code == 400
        bad = client.patch(
            f"/albums/{album_id}/people/{cluster_id}", json={"label": "x\n"}
        )
        assert bad.status_code == 400
        missing = client.patch(
            f"/albums/{album_id}/people/missing", json={"label": "Me"}
        )
        assert missing.status_code == 404
        refreshed = client.get(f"/albums/{album_id}/people")
        assert (
            next(
                c for c in refreshed.json()["clusters"] if c["cluster_id"] == cluster_id
            )["label"]
            == "Alex"
        )


def test_replacement_cannot_escape_original_subset(tmp_path):
    database, album_id, ids, _, _ = _named_album(tmp_path)
    result = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(
            album_id=album_id,
            prompt="选2张，至少2张有我",
            subset_photo_ids=[ids["c"], ids["f"]],
        )
    )
    replaced = ReplacementService(database, FakeSelectionProvider()).replace(
        result.selection_id, SelectionReplacementRequest(remove_photo_id=ids["c"])
    )
    assert not replaced.feasible  # e satisfies the quota but is outside the subset.


def test_replacement_revalidates_locked_quality(tmp_path):
    database, album_id, ids, _, _ = _named_album(tmp_path)
    result = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(album_id=album_id, prompt="选2张，至少2张有我")
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE photos SET auto_reject = 1 WHERE id = ?", (ids["f"],)
        )
    with pytest.raises(ValueError, match="locked photo quality changed"):
        ReplacementService(database, FakeSelectionProvider()).replace(
            result.selection_id, SelectionReplacementRequest(remove_photo_id=ids["c"])
        )


def test_solver_missing_never_drops_person_quota(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def no_ortools(name, *args, **kwargs):
        if name.startswith("ortools"):
            raise ImportError("test: ortools unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ortools)
    result = optimize_collection(
        [
            OptimizationCandidate(0, 1, "a", frozenset({"me"})),
        ],
        1,
        1,
        {"me": 1},
    )
    assert result.indices == []
    assert result.status == "person_quota_solver_unavailable"


@pytest.mark.parametrize("minimum", [True, "2", 2.5, 0, 51])
def test_structured_quota_requires_bounded_integer(minimum):
    with pytest.raises(ValueError):
        SelectionRequest(
            album_id="test", prompt="选2张", person_minimums={"Me": minimum}
        )
