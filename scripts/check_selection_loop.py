"""Real cached model scores + frozen OpenCLIP + explicitly synthetic proxy event.

Uses a SQLite online BACKUP. Never writes user feedback to the live database.
This is an engineering intervention test, NOT human aesthetic improvement.
"""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.aesthetics.provider import PyiqaMusiqProvider
from ai.aesthetics.service import AestheticsService
from ai.index.embedding import create_embedding_provider
from ai.preferences.memory import build_memory_evidence
from ai.preferences.repository import PreferenceRepository, PreferenceEvent
from ai.schemas import SelectionRequest
from ai.selection.service import SelectionService
from ai.storage import Database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "isolated.db"
    with (
        sqlite3.connect(
            args.source_db.resolve().as_uri() + "?mode=ro", uri=True
        ) as source,
        sqlite3.connect(target) as dest,
    ):
        source.backup(dest)
    db = Database(target)
    with db.connect() as c:
        album = c.execute("SELECT id FROM albums ORDER BY id LIMIT 1").fetchone()["id"]
    provider = create_embedding_provider(
        "openclip-multilingual", cache_dir=args.models, device="cpu"
    )
    assessment = AestheticsService(db, PyiqaMusiqProvider(args.models / "musiq"))
    service = SelectionService(db, provider, aesthetics_service=assessment)
    common = dict(
        album_id=album, prompt="选 4 张旅行照片", user_id="isolated-proxy-loop-20261005"
    )
    report = {
        "scope": "engineering-only; development album; synthetic intervention on isolated backup",
        "human_observed": False,
        "independent_preference_accuracy": None,
        "runs": {},
    }

    def run(label, **options):
        started = time.perf_counter()
        result = service.select(SelectionRequest(**common, **options))
        report["runs"][label] = {
            "seconds": time.perf_counter() - started,
            "response": result.model_dump(),
        }
        return result

    baseline = run("baseline")
    learned = run("learned_quality", use_learned_quality=True)
    assert baseline.feasible and learned.feasible and len(learned.selected) >= 2
    # Deliberately prefer the lower-ranked displayed item, to probe response to
    # an explicit intervention. This does NOT assert it is aesthetically better.
    winner, loser = learned.selected[-1].photo_id, learned.selected[0].photo_id
    with db.connect() as c:
        rows = {
            r["id"]: r
            for r in c.execute("SELECT * FROM photos WHERE album_id=?", (album,))
        }
    query = provider.embed_text(learned.query_text)
    event = PreferenceEvent(
        id="isolated-proxy-intervention-1",
        user_id=common["user_id"],
        album_id=album,
        selection_id=learned.selection_id,
        query_text=learned.query_text,
        preferred_photo_id=winner,
        rejected_photo_id=loser,
        choice="preferred",
        provider_fingerprint=provider.name,
        feature_schema="engineering-proxy-only",
        preferred_features=(0.0,),
        rejected_features=(0.0,),
        base_margin=0.0,
        model_id_at_display=None,
        context={
            "source": "assistant_proxy",
            "authorized_by_user": True,
            "human_observed": False,
            "label_policy": "counterfactual-lower-rank-intervention-not-aesthetic-judgment",
            "memory_evidence": build_memory_evidence(
                provider, query, rows[winner], rows[loser]
            ),
        },
    )
    PreferenceRepository(db).insert_event(event)
    excluded = run(
        "proxy_excluded_default", use_learned_quality=True, use_preference_memory=True
    )
    personalized = run(
        "explicit_proxy_memory",
        use_learned_quality=True,
        use_preference_memory=True,
        allow_proxy_memory=True,
    )
    reverted = run("memory_disabled_again", use_learned_quality=True)
    assert not excluded.preference_memory["applied"]
    assert (
        personalized.preference_memory["applied"]
        and event.id in personalized.preference_memory["event_ids"]
    )
    assert [p.model_dump() for p in learned.selected] == [
        p.model_dump() for p in reverted.selected
    ]
    report.update(
        status="passed",
        intervention={"preferred": winner, "rejected": loser},
        source_database_modified=False,
        trained=False,
    )
    (args.output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "runs": {
                    k: {
                        "seconds": round(v["seconds"], 3),
                        "selected": [p["filename"] for p in v["response"]["selected"]],
                    }
                    for k, v in report["runs"].items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
