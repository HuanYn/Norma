from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from ai import app as app_module
from ai.config import Settings
from ai.storage import Database


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(
            host="127.0.0.1",
            port=8765,
            data_dir=data_dir,
            log_level="INFO",
            embedding_provider="lightweight",
        ),
    )
    return TestClient(app_module.app)


def _album(path: Path, colors: list[tuple[int, int, int]]) -> None:
    path.mkdir()
    for index, color in enumerate(colors):
        Image.new("RGB", (320, 240), color).save(
            path / f"photo-{index}.jpg", "JPEG", quality=92
        )


def test_human_judgment_and_retrieval_evaluation_flow(
    tmp_path: Path, monkeypatch
) -> None:
    album = tmp_path / "album"
    _album(album, [(5, 10, 30), (15, 25, 70), (210, 190, 80), (35, 110, 55)])

    with _client(tmp_path, monkeypatch) as client:
        indexed = client.post("/albums/index", json={"folder": str(album)}).json()
        album_id = indexed["album_id"]
        assert client.post(f"/albums/{album_id}/embed").status_code == 200

        created = client.post(
            "/evaluation/queries",
            json={
                "album_id": album_id,
                "query_text": "  night   city  ",
                "notes": "manual test",
            },
        )
        assert created.status_code == 200, created.text
        query = created.json()
        assert query["query_text"] == "night city"
        assert query["judgment_count"] == 0

        unlabeled = client.get(
            f"/evaluation/queries/{query['id']}/candidates", params={"limit": 4}
        )
        assert unlabeled.status_code == 200, unlabeled.text
        candidates = unlabeled.json()["items"]
        assert [item["rank"] for item in candidates] == [1, 2, 3, 4]
        assert all(item["relevance"] is None for item in candidates)

        labels = [3, 0, 1, 0]
        judged = client.put(
            f"/evaluation/queries/{query['id']}/judgments",
            json={
                "annotator": "tester",
                "judgments": [
                    {"photo_id": item["photo_id"], "relevance": relevance}
                    for item, relevance in zip(candidates, labels, strict=True)
                ],
            },
        )
        assert judged.status_code == 200, judged.text
        assert judged.json() == {
            "query_id": query["id"],
            "upserted_count": 4,
            "judgment_count": 4,
            "relevant_count": 2,
        }

        second = client.post(
            "/evaluation/queries",
            json={"album_id": album_id, "query_text": "green nature"},
        ).json()
        report = client.post(
            f"/albums/{album_id}/evaluation/runs",
            json={"cutoffs": [4, 1, 2, 2]},
        )
        assert report.status_code == 200, report.text
        payload = report.json()
        assert payload["cutoffs"] == [1, 2, 4]
        assert payload["query_count"] == 1
        assert payload["skipped_query_count"] == 1
        assert payload["macro_mrr"] == 1.0
        assert payload["macro_precision_at"] == {
            "1": 1.0,
            "2": 0.5,
            "4": 0.5,
        }
        assert payload["macro_recall_at"] == {
            "1": 0.5,
            "2": 0.5,
            "4": 1.0,
        }
        assert payload["macro_ndcg_at"]["1"] == 1.0
        assert payload["provider"] == "lightweight-semantic-v1"

        with app_module.database.connect() as connection:
            persisted = connection.execute(
                "SELECT result_json FROM evaluation_runs WHERE id = ?",
                (payload["run_id"],),
            ).fetchone()
        assert persisted is not None
        stored_report = client.get(f"/evaluation/runs/{payload['run_id']}")
        assert stored_report.status_code == 200
        assert stored_report.json() == payload
        assert payload["queries"][0]["relevance_by_photo"] == {
            item["photo_id"]: relevance
            for item, relevance in zip(candidates, labels, strict=True)
        }

        listed = client.get(f"/albums/{album_id}/evaluation/queries").json()
        assert listed["total"] == 2
        listed_by_id = {item["id"]: item for item in listed["items"]}
        assert listed_by_id[query["id"]]["judgment_count"] == 4
        assert listed_by_id[second["id"]]["judgment_count"] == 0


def test_evaluation_rejects_duplicates_cross_album_photos_and_stale_cache(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _album(first, [(5, 10, 30)])
    _album(second, [(210, 190, 80)])

    with _client(tmp_path, monkeypatch) as client:
        first_index = client.post("/albums/index", json={"folder": str(first)}).json()
        second_index = client.post("/albums/index", json={"folder": str(second)}).json()
        assert (
            client.post(f"/albums/{first_index['album_id']}/embed").status_code == 200
        )
        query_request = {
            "album_id": first_index["album_id"],
            "query_text": "night",
        }
        query = client.post("/evaluation/queries", json=query_request).json()
        duplicate = client.post("/evaluation/queries", json=query_request)
        assert duplicate.status_code == 409

        cross_album = client.put(
            f"/evaluation/queries/{query['id']}/judgments",
            json={
                "judgments": [
                    {"photo_id": second_index["photos"][0]["id"], "relevance": 2}
                ]
            },
        )
        assert cross_album.status_code == 400

        Image.new("RGB", (320, 240), (250, 250, 250)).save(
            first / "photo-0.jpg", "JPEG", quality=92
        )
        stale = client.get(f"/evaluation/queries/{query['id']}/candidates")
        assert stale.status_code == 404
