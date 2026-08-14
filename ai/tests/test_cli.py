from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from ai.cli import main


def _jpeg(path: Path, color: tuple[int, int, int], offset: int) -> None:
    image = Image.new("RGB", (520, 360), color)
    draw = ImageDraw.Draw(image)
    for x in range(20 + offset, 500, 55):
        draw.rectangle((x, 30, x + 18, 330), fill=(220, 175, 80))
    image.save(path, "JPEG", quality=93)


def _result(capsys) -> object:
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    return payload["result"]


def test_python_cli_runs_prepare_search_and_selection(tmp_path: Path, capsys) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _jpeg(album / "night-1.jpg", (5, 12, 35), 0)
    _jpeg(album / "night-2.jpg", (12, 24, 60), 5)
    _jpeg(album / "bright.jpg", (220, 190, 95), 10)
    data_dir = tmp_path / "data"

    assert (
        main(
            [
                "--data-dir",
                str(data_dir),
                "prepare",
                str(album),
                "--skip-people",
            ]
        )
        == 0
    )
    prepared = _result(capsys)
    album_id = prepared["album"]["album_id"]
    assert prepared["album"]["total"] == 3
    assert prepared["embedding"]["count"] == 3

    assert main(["--data-dir", str(data_dir), "albums"]) == 0
    albums = _result(capsys)
    assert albums[0]["id"] == album_id
    assert albums[0]["photos"] == 3

    assert (
        main(["--data-dir", str(data_dir), "photos", album_id, "--include-rejects"])
        == 0
    )
    photos = _result(capsys)
    assert len(photos) == 3
    assert all(photo["embedded"] for photo in photos)

    assert (
        main(["--data-dir", str(data_dir), "search", album_id, "night", "--limit", "2"])
        == 0
    )
    search = _result(capsys)
    assert search["mode"] == "text"
    assert len(search["matches"]) == 2

    assert (
        main(
            [
                "--data-dir",
                str(data_dir),
                "select",
                album_id,
                "Select 1 photo of night, include blurry",
            ]
        )
        == 0
    )
    selection = _result(capsys)
    assert selection["feasible"]
    assert len(selection["selected"]) == 1

    assert main(["--data-dir", str(data_dir), "album", album_id]) == 0
    detail = _result(capsys)
    assert detail["photo_count"] == 3
    assert detail["embedded_count"] == 3
    assert detail["embedding_provider"] == "lightweight-semantic-v1"
    assert detail["selection_count"] == 1

    assert main(["--data-dir", str(data_dir), "selection-history", album_id]) == 0
    history = _result(capsys)
    assert history["total"] == 1
    assert history["items"][0]["selected_count"] == 1

    assert main(["--data-dir", str(data_dir), "jobs"]) == 0
    jobs = _result(capsys)
    assert jobs["total"] == 0


def test_python_cli_reports_domain_errors_as_json(tmp_path: Path, capsys) -> None:
    exit_code = main(["--data-dir", str(tmp_path / "data"), "embed", "missing-album"])
    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert "not found" in payload["error"]
