"""Local, score-blind pair collection. Labels never train or change selections.

The protocol is frozen before annotation; labels are frozen before prediction.
Human-pilot labels do not acquire independent-human status automatically.
"""

import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Literal

from ai.selection.evaluation import evaluate


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build_bundle(manifest: dict, destination: Path):
    """Explicit CLI input only. Serve copied derivatives, never arbitrary paths."""
    from PIL import Image, ImageOps

    protocol = {
        k: manifest[k]
        for k in [
            "method_revision",
            "development_and_memory_sha256",
            "development_and_memory_groups",
            "label_provenance",
        ]
    }
    if protocol["label_provenance"] not in {
        "human-pilot",
        "independent-human",
        "assistant-proxy",
        "controlled-fixture",
    }:
        raise ValueError("Explicit label provenance required")
    if (
        protocol["label_provenance"] == "independent-human"
        and manifest.get("independent_sources_verified") is not True
    ):
        raise ValueError("Independent source audit required; otherwise use human-pilot")
    protocol.update(
        schema="norma-review-protocol-v1",
        cases=[],
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=42,
    )
    inputs = {}
    groups = {}
    pair_keys = set()
    for item in manifest["cases"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", item["id"]):
            raise ValueError("Invalid case id")
        case = {k: item[k] for k in ["id", "group", "query"]}
        if not isinstance(case["query"], str) or not 1 <= len(case["query"]) <= 1000:
            raise ValueError("Invalid query")
        for side in ["left", "right"]:
            path = Path(item[side]).resolve(strict=True)
            data = path.read_bytes()
            if len(data) > 32 * 1024 * 1024:
                raise ValueError("Image exceeds 32MiB budget")
            sha = digest(data)
            case[side + "_sha256"] = sha
            inputs[sha] = data
            if sha in groups and groups[sha] != case["group"]:
                raise ValueError("Photo appears in multiple source groups")
            groups[sha] = case["group"]
        pair_key = (
            tuple(sorted([case["left_sha256"], case["right_sha256"]])),
            case["query"].strip(),
        )
        if pair_key in pair_keys:
            raise ValueError("Duplicate pair/query")
        pair_keys.add(pair_key)
        protocol["cases"].append(case)
    # Reuse the independently implemented leakage/schema checks, not model labels.
    document = protocol | {
        "schema": "norma-independent-pairs-v1",
        "labels_frozen_before_predictions": True,
        "label_manifest_sha256": "preflight",
        "cases": [c | {"winner": "tie"} for c in protocol["cases"]],
    }
    scores = {c["id"]: [0, 0] for c in protocol["cases"]}
    evaluate(document, {"baseline": scores, "fixture": scores}, resamples=100)
    random.Random(42).shuffle(protocol["cases"])
    rng = random.Random(43)
    for case in protocol["cases"]:
        case["swapped"] = bool(rng.randrange(2))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "images").mkdir()
    import io

    protocol["image_derivative_sha256"] = {}
    for sha, data in inputs.items():
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1200, 1200))
            target = destination / "images" / (sha + ".jpg")
            image.save(target, "JPEG", quality=90)
            protocol["image_derivative_sha256"][sha] = digest(target.read_bytes())
    raw = canonical(protocol).encode()
    (destination / "protocol.json").write_bytes(raw)
    with sqlite3.connect(destination / "labels.sqlite") as c:
        c.executescript(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE labels (case_id TEXT PRIMARY KEY,winner TEXT NOT NULL,revision INTEGER NOT NULL); CREATE TABLE events (id INTEGER PRIMARY KEY,case_id TEXT,winner TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP);"
        )
        c.execute("INSERT INTO metadata VALUES (?,?)", ("protocol_sha256", digest(raw)))
    return protocol


class ReviewStore:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)

    def connect(self):
        c = sqlite3.connect(
            f"file:{(self.root / 'labels.sqlite').as_posix()}?mode=rw",
            uri=True,
            timeout=10,
        )
        c.row_factory = sqlite3.Row
        return c

    def protocol(self, c):
        raw = (self.root / "protocol.json").read_bytes()
        expected = c.execute(
            "SELECT value FROM metadata WHERE key='protocol_sha256'"
        ).fetchone()[0]
        if digest(raw) != expected:
            raise ValueError("Frozen protocol changed")
        return json.loads(raw)

    def state(self):
        with self.connect() as c:
            p = self.protocol(c)
            labels = {r["case_id"]: dict(r) for r in c.execute("SELECT * FROM labels")}
            frozen = c.execute(
                "SELECT value FROM metadata WHERE key='frozen_labels'"
            ).fetchone()
        return {
            "total": len(p["cases"]),
            "answered": len(labels),
            "frozen": bool(frozen),
            "provenance": p["label_provenance"],
            "method_revision": p["method_revision"],
            "cases": [
                {
                    "id": v["id"],
                    "query": v["query"],
                    "a": "/images/"
                    + v["right_sha256" if v["swapped"] else "left_sha256"]
                    + ".jpg",
                    "b": "/images/"
                    + v["left_sha256" if v["swapped"] else "right_sha256"]
                    + ".jpg",
                    "answer": labels.get(v["id"], {}).get("winner"),
                    "revision": labels.get(v["id"], {}).get("revision", 0),
                }
                for v in p["cases"]
            ],
        }

    def answer(self, case_id, winner, revision):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            p = self.protocol(c)
            if c.execute("SELECT 1 FROM metadata WHERE key='frozen_labels'").fetchone():
                raise ValueError("Labels are frozen; start a new review to change them")
            if case_id not in {v["id"] for v in p["cases"]}:
                raise ValueError("Unknown case")
            row = c.execute(
                "SELECT revision FROM labels WHERE case_id=?", (case_id,)
            ).fetchone()
            if (row[0] if row else 0) != revision:
                raise ValueError("Stale label revision; refresh before saving")
            if winner not in {"a", "b", "tie"}:
                raise ValueError("Choose A, B or tie; skip stays unanswered")
            c.execute(
                "INSERT OR REPLACE INTO labels VALUES (?,?,?)",
                (case_id, winner, revision + 1),
            )
            c.execute(
                "INSERT INTO events(case_id,winner) VALUES (?,?)", (case_id, winner)
            )

    def freeze(self, human_confirmed):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            p = self.protocol(c)
            old = c.execute(
                "SELECT value FROM metadata WHERE key='frozen_labels'"
            ).fetchone()
            if old:
                return json.loads(old[0])
            rows = {
                r["case_id"]: r["winner"] for r in c.execute("SELECT * FROM labels")
            }
            if set(rows) != {v["id"] for v in p["cases"]}:
                raise ValueError(
                    "Every pair needs a label before freezing; skipped pairs are not ties"
                )
            human = p["label_provenance"] in {"human-pilot", "independent-human"}
            if human and not human_confirmed:
                raise ValueError("Confirm these are your own visual judgments")
            cases = []
            for v in p["cases"]:
                answer = rows[v["id"]]
                winner = (
                    "tie"
                    if answer == "tie"
                    else ("right" if (answer == "a") == v["swapped"] else "left")
                )
                cases.append(
                    {
                        k: v[k]
                        for k in ["id", "group", "query", "left_sha256", "right_sha256"]
                    }
                    | {"winner": winner}
                )
            doc = {
                k: p[k]
                for k in [
                    "method_revision",
                    "development_and_memory_sha256",
                    "development_and_memory_groups",
                    "label_provenance",
                ]
            }
            doc.update(
                schema="norma-independent-pairs-v1",
                labels_frozen_before_predictions=True,
                human_observed=human,
                frozen_at=datetime.now(timezone.utc).isoformat(),
                protocol_sha256=digest(canonical(p).encode()),
                cases=cases,
            )
            c.execute(
                "INSERT INTO metadata VALUES (?,?)", ("frozen_labels", canonical(doc))
            )
            return doc

    def export(self):
        with self.connect() as c:
            self.protocol(c)
            row = c.execute(
                "SELECT value FROM metadata WHERE key='frozen_labels'"
            ).fetchone()
            if not row:
                raise ValueError("Freeze labels before export")
            return row[0].encode()


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    winner: Literal["a", "b", "tie"]
    revision: int


class Freeze(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    human_confirmed: bool = False


def create_review_app(root: Path):
    store = ReviewStore(root)
    app = FastAPI(docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    async def body(request, cls):
        from ai.exploration.server import json_body

        if request.headers.get("x-norma-review") != "1" or request.headers.get(
            "origin", str(request.base_url).rstrip("/")
        ) != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Same-origin review required")
        return await json_body(request, cls, 2048)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(
            (
                Path(__file__).resolve().parents[2]
                / "scripts/templates/pair_review.html"
            ).read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/state")
    def state():
        try:
            return store.state()
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/images/{name}")
    def image(name: str):
        if not re.fullmatch("[a-f0-9]{64}\\.jpg", name):
            raise HTTPException(404)
        try:
            with store.connect() as c:
                p = store.protocol(c)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        if name[:-4] not in p["image_derivative_sha256"]:
            raise HTTPException(404)
        target = store.root / "images" / name
        if not target.is_file():
            raise HTTPException(409, "Frozen image missing")
        if digest(target.read_bytes()) != p["image_derivative_sha256"][name[:-4]]:
            raise HTTPException(409, "Image changed")
        return FileResponse(target, headers={"Cache-Control": "no-store"})

    @app.post("/labels/{case_id}")
    async def answer(case_id: str, request: Request):
        payload = await body(request, Answer)
        try:
            store.answer(case_id, payload.winner, payload.revision)
            return store.state()
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/freeze")
    async def freeze(request: Request):
        payload = await body(request, Freeze)
        try:
            store.freeze(payload.human_confirmed)
            return store.state()
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/labels.json")
    def export():
        from fastapi import Response

        try:
            return Response(
                store.export(),
                media_type="application/json",
                headers={
                    "Content-Disposition": 'attachment; filename="labels.json"',
                    "Cache-Control": "no-store",
                },
            )
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    return app
