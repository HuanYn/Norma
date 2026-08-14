from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from ai import app as app_module
from ai.storage import Database


def test_built_web_app_and_api_share_one_origin(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "database", Database(tmp_path / "norma.db"))

    with TestClient(app_module.app) as client:
        page = client.get("/")
        health = client.get("/health")

        asset_path = re.search(r'src="([^"]+\.js)"', page.text)
        assert asset_path is not None
        asset = client.get(asset_path.group(1))

    assert page.status_code == 200
    assert "Norma" in page.text
    assert page.headers["content-type"].startswith("text/html")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert asset.status_code == 200
    assert "javascript" in asset.headers["content-type"]
