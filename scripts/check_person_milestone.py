"""Run two isolated before/after engineering probes, not recognition benchmarks.

The baseline executes pinned historical modules with today's shared storage and
schema dependencies. All generated photos and SQLite databases stay inside an
automatically cleaned temporary directory. No server, cloud, or user data access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_COMMIT = "5a28460a2f72c80c0b5460351242018ed5e42328"
MODULES = (
    "ai/people/indexer.py",
    "ai/selection/parser.py",
    "ai/selection/optimizer.py",
    "ai/selection/service.py",
)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"controlled probe failed: {message}")


def _legacy_module(path: str) -> tuple[types.ModuleType, str]:
    source = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    name = "_norma_person_probe_legacy_" + path.replace("/", "_").replace(".", "_")
    module = types.ModuleType(name)
    module.__file__ = f"git:{BASELINE_COMMIT}:{path}"
    # dataclasses resolves module globals via sys.modules during class creation.
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, _digest(source)


def run_comparison() -> dict[str, object]:
    from ai.index import AlbumIndexer
    from ai.people.indexer import PeopleIndexer
    from ai.people.labels import PersonLabelService
    from ai.schemas import SelectionRequest
    from ai.selection.parser import parse_selection_prompt
    from ai.selection.service import SelectionService
    from ai.storage import Database
    from ai.tests.test_people_labels import LabelFaceProvider, _image
    from ai.tests.test_person_selection import _named_album
    from ai.tests.test_selection import FakeSelectionProvider

    legacy = {}
    baseline_hashes = {}
    for path in MODULES:
        module, source_hash = _legacy_module(path)
        legacy[path] = module
        baseline_hashes[path] = source_hash
    old_people = legacy["ai/people/indexer.py"]
    old_parser = legacy["ai/selection/parser.py"]
    old_optimizer = legacy["ai/selection/optimizer.py"]
    old_selection = legacy["ai/selection/service.py"]
    # Bind the baseline service to its actual historical parser and optimizer.
    old_selection.parse_selection_prompt = old_parser.parse_selection_prompt
    old_selection.has_semantic_content = old_parser.has_semantic_content
    old_selection.optimize_collection = old_optimizer.optimize_collection
    old_selection.OptimizationCandidate = old_optimizer.OptimizationCandidate

    with tempfile.TemporaryDirectory(prefix="norma-person-probe-") as scratch:
        root = Path(scratch)
        lifecycle = {}
        for variant, indexer_type in (
            ("before", old_people.PeopleIndexer),
            ("after", PeopleIndexer),
        ):
            case_root = root / variant
            album = case_root / "album"
            album.mkdir(parents=True)
            for filename in ("a1.jpg", "a2.jpg"):
                _image(album / filename)
            data_dir = case_root / "data"
            database = Database(data_dir / "norma.db")
            album_id = AlbumIndexer(database, data_dir).index(album).album_id
            indexer = indexer_type(database, data_dir, LabelFaceProvider())
            initial = indexer.index(album_id)
            cluster_id = initial.clusters[0].cluster_id
            if variant == "before":
                # The baseline had no naming API. Seed its existing label column
                # directly, solely to test whether reindex destroys a stored name.
                with database.connect() as connection:
                    connection.execute(
                        "UPDATE person_clusters SET label = 'Me' WHERE id = ?",
                        (cluster_id,),
                    )
            else:
                PersonLabelService(database).set(album_id, cluster_id, "Me")
            before = indexer.get(album_id)
            refreshed = indexer.index(album_id)
            cluster = refreshed.clusters[0]
            lifecycle[variant] = {
                "source_image_sha256": {
                    path.name: _digest(path.read_bytes())
                    for path in sorted(album.glob("*.jpg"))
                },
                "label_before_reindex": before.clusters[0].label,
                "label_after_reindex": cluster.label,
                "confirmed_status_available": variant == "after",
                "label_status_after": cluster.label_status
                if variant == "after"
                else None,
                "photos": 2,
                "faces_after": refreshed.total_faces,
                "clusters_after": refreshed.cluster_count,
                "computed_after": refreshed.computed_count,
                "reused_after": refreshed.reused_count,
                "named_clusters_preserved": int(cluster.label == "Me"),
                "named_clusters_expected": 1,
            }
        _require(
            lifecycle["before"]["source_image_sha256"]
            == lifecycle["after"]["source_image_sha256"],
            "lifecycle inputs differ",
        )

        quota_root = root / "quota"
        quota_root.mkdir()
        database, album_id, ids, _, _ = _named_album(quota_root)
        person_ids = {ids["c"], ids["e"], ids["f"]}
        request = SelectionRequest(
            album_id=album_id, prompt="选2张", person_minimums={"Me": 2}
        )
        quota = {}
        for variant, service_type in (
            ("before", old_selection.SelectionService),
            ("after", SelectionService),
        ):
            result = service_type(
                database, FakeSelectionProvider(), preference_mode="record-only"
            ).select(request)
            present = sum(photo.photo_id in person_ids for photo in result.selected)
            quota[variant] = {
                "selected": [
                    {
                        "filename": photo.filename,
                        "score": photo.total_score,
                        "contains_me": photo.photo_id in person_ids,
                    }
                    for photo in result.selected
                ],
                "selected_count": len(result.selected),
                "person_photo_count": present,
                "person_photo_minimum": 2,
                "person_quota_satisfied": present >= 2,
                "requested_person_minimums": request.person_minimums,
                "returned_person_minimums": result.constraints.person_minimums,
                "feasible_flag": result.feasible,
                "solver": result.solver,
                "solver_status": result.solver_status,
            }
        with database.connect() as connection:
            candidate_rows = connection.execute(
                "SELECT id, quality_score, auto_reject, similarity_group, absolute_path FROM photos WHERE album_id = ? ORDER BY absolute_path",
                (album_id,),
            ).fetchall()
        candidates = [
            {
                "filename": Path(row["absolute_path"]).name,
                "quality_score": row["quality_score"],
                "auto_reject": bool(row["auto_reject"]),
                "similarity_group": row["similarity_group"],
                "contains_me": row["id"] in person_ids,
                "source_sha256": _digest(Path(row["absolute_path"]).read_bytes()),
            }
            for row in candidate_rows
        ]

    _require(
        lifecycle["before"]["label_after_reindex"] == "Unknown",
        "baseline name loss was not reproduced",
    )
    _require(
        lifecycle["after"]["label_after_reindex"] == "Me",
        "current name preservation failed",
    )
    _require(
        quota["before"]["person_photo_count"] == 1, "baseline fixture selection changed"
    )
    _require(quota["after"]["person_photo_count"] == 2, "current person count is wrong")
    _require(
        quota["after"]["person_quota_satisfied"], "current person quota not satisfied"
    )
    prompt = "选2张，至少2张有我"
    return {
        "schema_version": 1,
        "report_type": "controlled_engineering_regression_not_model_benchmark",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "baseline_commit": BASELINE_COMMIT,
        "baseline_module_sha256": baseline_hashes,
        "current_module_sha256": {
            path: _digest((PROJECT_ROOT / path).read_bytes())
            for path in (
                *MODULES,
                "ai/people/labels.py",
                "ai/selection/people_constraints.py",
                "ai/storage.py",
                "ai/schemas.py",
            )
        },
        "fixture_module_sha256": {
            path: _digest((PROJECT_ROOT / path).read_bytes())
            for path in (
                "ai/tests/test_people_labels.py",
                "ai/tests/test_person_selection.py",
                "ai/tests/test_selection.py",
            )
        },
        "isolation": "TemporaryDirectory; separate lifecycle databases; one shared isolated quota fixture; removed after run",
        "fixture_source": "deterministic synthetic JPEGs and fake providers from ai/tests; no real faces, private photos, cloud calls or model training",
        "method_limits": [
            "Historical selected modules execute against current shared schema/storage/dependencies, not a full historical environment reconstruction.",
            "Before naming is seeded with direct SQL because the baseline had no naming API; after naming uses PersonLabelService.",
            "Quota probe passes a current typed request carrying an explicit person_minimums field into both services; the historical service does not consume that field. It is not a claim that the old public API supported person quotas.",
            "Only two engineering cases are compared. These are not face accuracy, generalization, aesthetic quality, preference learning or latency benchmarks.",
            "The exclusion and face identity are synthetic fixtures, not automatically annotated real people.",
        ],
        "cases": {
            "unchanged_reindex_preserves_named_identity": lifecycle,
            "person_quota_two_of_two": {"shared_candidates": candidates, **quota},
        },
        "parser_probe": {
            "prompt": prompt,
            "before_person_minimums": getattr(
                old_parser.parse_selection_prompt(prompt), "person_minimums", {}
            ),
            "after_person_minimums": parse_selection_prompt(prompt).person_minimums,
        },
        "assertions_passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="optional new JSON file; refuses overwrite"
    )
    args = parser.parse_args(argv)
    report = run_comparison()
    # ASCII JSON stays round-trippable on Windows consoles regardless of codepage.
    text = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
