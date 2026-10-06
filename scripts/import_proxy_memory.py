"""Import previously inspected, user-authorized proxy pairs as case memory only.

No training, no human-feedback claim, no inferred labels. Existing images and
embeddings must match the source pins. Explicit --apply is required for writes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.index.embedding import create_embedding_provider
from ai.preferences.memory import build_memory_evidence, _load_snapshot_photo
from ai.preferences.repository import PreferenceEvent, PreferenceRepository
from ai.storage import Database
from scripts.seed_proxy_preferences import load_proxy_preferences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--album-id", required=True)
    p.add_argument("--models", type=Path, default=Path(".norma/data/models"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    a.database.resolve(strict=True)
    if a.output.exists():
        p.error("Use a new output directory to preserve receipts")
    dataset = load_proxy_preferences(allow_assistant_proxy=True)
    db = Database(a.database)
    provider = create_embedding_provider(
        "openclip-multilingual", cache_dir=a.models.resolve(), device="cpu"
    )
    with db.connect() as c:
        rows = {
            r["embedding_source_sha256"]: r
            for r in c.execute("SELECT * FROM photos WHERE album_id=?", (a.album_id,))
        }
        existing = {r[0] for r in c.execute("SELECT id FROM preference_events")}
    events = []
    for pair in dataset:
        left, right = (
            rows.get(pair["left_asset"]["sha256"]),
            rows.get(pair["right_asset"]["sha256"]),
        )
        if left is None or right is None:
            raise ValueError(
                "Import album must contain both byte-pinned photos for every pair"
            )
        preferred, rejected = (
            (right, left)
            if pair["choice"] == "preferred"
            and pair["preferred_file"] == pair["right_file"]
            else (left, right)
        )
        query = provider.embed_text(pair["query_text"])
        evidence = build_memory_evidence(provider, query, preferred, rejected)
        pv, _ = _load_snapshot_photo(preferred, provider)
        rv, _ = _load_snapshot_photo(rejected, provider)
        identity = "|".join(
            [pair["dataset_sha256"], pair["id"], a.album_id, provider.name, "local"]
        )
        events.append(
            PreferenceEvent(
                id="proxy-memory-" + hashlib.sha256(identity.encode()).hexdigest(),
                user_id="local",
                album_id=a.album_id,
                selection_id=None,
                query_text=pair["query_text"],
                preferred_photo_id=preferred["id"],
                rejected_photo_id=rejected["id"],
                choice=pair["choice"],
                provider_fingerprint=provider.name,
                feature_schema="proxy-frozen-openclip-memory-only-v1",
                preferred_features=tuple(float(v) for v in pv),
                rejected_features=tuple(float(v) for v in rv),
                base_margin=float(query @ (pv - rv)),
                model_id_at_display=None,
                context={
                    "source": "assistant_proxy",
                    "authorized_by_user": True,
                    "human_observed": False,
                    "dataset_sha256": pair["dataset_sha256"],
                    "pair_id": pair["id"],
                    "rationale": pair["rationale"],
                    "limitation": pair["limitation"],
                    "usage": "development_proxy_only",
                    "evaluation": pair["evaluation"],
                    "memory_evidence": evidence,
                    "evidence_captured_at_import_not_historical_display": True,
                },
            )
        )
    a.output.mkdir(parents=True)
    inserted = []
    if a.apply:
        with (
            sqlite3.connect(
                a.database.resolve().as_uri() + "?mode=ro", uri=True
            ) as source,
            sqlite3.connect(a.output / "before.sqlite") as dest,
        ):
            source.backup(dest)
        repo = PreferenceRepository(db)
        for event in events:
            if event.id not in existing:
                repo.insert_event(event)
                inserted.append(event.id)
    report = dict(
        status="applied" if a.apply else "validated-only",
        pairs=len(events),
        inserted=inserted,
        already_present=sum(e.id in existing for e in events),
        human_observed=False,
        trained=False,
        clear_winners=sum(e.choice == "preferred" for e in events),
        ties=sum(e.choice == "tie" for e in events),
        evaluation_boundary="assistant-proxy development memory, NOT human preference improvement",
    )
    (a.output / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
