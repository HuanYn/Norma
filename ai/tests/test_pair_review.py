import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from ai.selection.review import build_bundle, ReviewStore, create_review_app


def manifest(tmp_path, provenance="controlled-fixture"):
    paths = []
    for i in range(4):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (50, 50), (i * 50, 100, 20)).save(p)
        paths.append(str(p))
    return dict(
        method_revision="controlled-test-not-model",
        label_provenance=provenance,
        development_and_memory_sha256=[],
        development_and_memory_groups=[],
        cases=[
            dict(
                id="one",
                group="first",
                query="选适合旅行分享的照片",
                left=paths[0],
                right=paths[1],
            ),
            dict(
                id="two",
                group="second",
                query="选喜欢的照片",
                left=paths[2],
                right=paths[3],
            ),
        ],
    )


def test_blind_state_no_paths_scores_and_freeze(tmp_path):
    p = build_bundle(manifest(tmp_path), tmp_path / "bundle")
    store = ReviewStore(tmp_path / "bundle")
    state = store.state()
    assert state["answered"] == 0
    assert all(
        not any(k in c for k in ["left_sha256", "winner", "score", "path", "swapped"])
        for c in state["cases"]
    )
    with pytest.raises(ValueError, match="Every pair"):
        store.freeze(False)
    with pytest.raises(ValueError, match="Freeze labels"):
        store.export()
    for c in state["cases"]:
        store.answer(c["id"], "a", 0)
    frozen = store.freeze(False)
    data = store.export()
    for c in p["cases"]:
        label = next(v for v in frozen["cases"] if v["id"] == c["id"])
        assert label["winner"] == ("right" if c["swapped"] else "left")
    assert not frozen["human_observed"]
    assert store.freeze(False) == frozen
    assert ReviewStore(tmp_path / "bundle").export() == data
    with pytest.raises(ValueError, match="frozen"):
        store.answer("one", "b", 1)


def test_revision_and_tie_skip_distinct(tmp_path):
    build_bundle(manifest(tmp_path), tmp_path / "bundle")
    store = ReviewStore(tmp_path / "bundle")
    store.answer("one", "a", 0)
    with pytest.raises(ValueError, match="Stale"):
        store.answer("one", "b", 0)
    with pytest.raises(ValueError, match="skip"):
        store.answer("two", "skip", 0)
    store.answer("one", "tie", 1)
    store.answer("two", "b", 0)
    assert (
        next(c for c in store.freeze(False)["cases"] if c["id"] == "one")["winner"]
        == "tie"
    )


def test_human_requires_confirmation_and_provenance_cannot_upgrade(tmp_path):
    build_bundle(manifest(tmp_path, "human-pilot"), tmp_path / "bundle")
    store = ReviewStore(tmp_path / "bundle")
    store.answer("one", "a", 0)
    store.answer("two", "b", 0)
    with pytest.raises(ValueError, match="own visual"):
        store.freeze(False)
    result = store.freeze(True)
    assert result["human_observed"]
    assert result["label_provenance"] == "human-pilot"


@pytest.mark.parametrize("failure", ["hash", "group", "duplicate", "audit"])
def test_contaminated_or_unverified_protocol_rejected(tmp_path, failure):
    m = manifest(tmp_path)
    if failure == "hash":
        m["development_and_memory_sha256"] = [
            hashlib.sha256(Path(m["cases"][0]["left"]).read_bytes()).hexdigest()
        ]
    if failure == "group":
        m["development_and_memory_groups"] = ["first"]
    if failure == "duplicate":
        m["cases"].append(m["cases"][0] | {"id": "three"})
    if failure == "audit":
        m["label_provenance"] = "independent-human"
    with pytest.raises(ValueError):
        build_bundle(m, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_protocol_tamper_and_api_security(tmp_path):
    build_bundle(manifest(tmp_path), tmp_path / "bundle")
    with TestClient(create_review_app(tmp_path / "bundle")) as client:
        assert client.get("/").status_code == 200
        state = client.get("/state").json()
        assert client.get(state["cases"][0]["a"]).status_code == 200
        assert client.get("/images/missing.jpg").status_code == 404
        assert (
            client.post("/labels/one", json={"winner": "a", "revision": 0}).status_code
            == 403
        )
        h = {"X-Norma-Review": "1"}
        assert (
            client.post(
                "/labels/one",
                json={"winner": "a", "revision": 0},
                headers=h | {"Origin": "https://other.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/labels/one", json={"winner": "a", "revision": 0}, headers=h
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/freeze", json={"human_confirmed": False}, headers=h
            ).status_code
            == 409
        )
        assert client.get("/labels.json").status_code == 409
        path = tmp_path / "bundle/protocol.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["cases"][0]["query"] = "changed"
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert client.get("/state").status_code == 409
        assert client.get(state["cases"][0]["a"]).status_code == 409
