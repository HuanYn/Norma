"""Real local curation regression; original photos read-only, no new model/GPU jobs."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import urllib.request


def api(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8767" + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    album = "998eaeb56c485567aa46f7753fbe6fc2"
    with sqlite3.connect(
        (Path(__file__).resolve().parents[1] / ".norma/demo-data/norma.db").as_uri() + "?mode=ro", uri=True
    ) as db:
        before = json.loads(
            db.execute(
                "SELECT result_json FROM workflow_curations_v1 WHERE id=?",
                ("aea69fb71b5d44eba7527725ad4fe50c",),
            ).fetchone()[0]
        )
        paths = [
            Path(r[0])
            for r in db.execute(
                "SELECT absolute_path FROM photos WHERE album_id=?", (album,)
            )
        ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    start = time.perf_counter()
    after = api(f"/workflow/albums/{album}/curations", {})
    elapsed = time.perf_counter() - start
    families = {
        "peak_close": [
            "IMG_2134.JPG",
            "IMG_2133.JPG",
            "IMG_2132.JPG",
            "IMG_2135.JPG",
            "IMG_2118.JPG",
            "IMG_2130.JPG",
        ],
        "pagoda_valley": ["IMG_2002.JPG", "IMG_2000.JPG", "IMG_1998.JPG"],
        "peak_river_wide": ["IMG_2128.JPG", "IMG_2129.JPG"],
        "lake_trees": ["IMG_2026.JPG", "IMG_2028.JPG"],
    }
    counts = {
        label: {
            "before": sum(p["filename"] in names for p in before["kept"]),
            "after": sum(p["filename"] in names for p in after["kept"]),
        }
        for label, names in families.items()
    }
    assert all(c["after"] == 1 for c in counts.values()), counts
    assert {
        "IMG_2134.JPG",
        "IMG_2128.JPG",
        "Norma-demo-temple.jpg",
        "Norma-demo-mountain.jpg",
    } <= {p["filename"] for p in after["kept"]}
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()
    )
    # Read-back rechecks current data. Content-bound spatial cache must be stable.
    start = time.perf_counter()
    checked = api("/workflow/curations/" + after["id"])
    warm = time.perf_counter() - start
    assert checked["snapshot_sha256"] == after["snapshot_sha256"]
    draft = api(f"/workflow/albums/{album}/selection-draft")
    assert set(draft["required_photo_ids"]) <= {p["photo_id"] for p in after["kept"]}
    draft = api(
        f"/workflow/albums/{album}/selection-draft",
        {k: v for k, v in draft.items() if k != "revision"}
        | {"curation_id": after["id"]},
    )
    result = {
        "status": "passed",
        "scope": "development album and user-identified composition families, not independent duplicate accuracy",
        "before": before,
        "after": after,
        "families": counts,
        "cold_seconds": elapsed,
        "warm_seconds": warm,
        "source_files_unchanged": len(hashes),
        "draft": draft,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "families",
                    "cold_seconds",
                    "warm_seconds",
                    "source_files_unchanged",
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
