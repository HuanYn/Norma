"""Prepare score-blind human pilot pairs, excluding known development photos.

All candidates remain one conservative source group. This is NOT a claim of
independent sampling or multi-user preference improvement. Never adds labels.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.index.similarity import perceptual_hash, difference_hash, hamming_distance
from ai.selection.review import build_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--development-db", action="append", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=12)
    args = p.parse_args()
    if args.output.exists() or not 2 <= args.pairs <= 50:
        p.error("New output directory and 2..50 pairs required")
    source = args.source.resolve(strict=True)
    known_paths, known_sha, known_hash = set(), set(), []
    for db in args.development_db:
        with sqlite3.connect(
            f"file:{db.resolve(strict=True).as_posix()}?mode=ro", uri=True
        ) as c:
            columns = {r[1] for r in c.execute("PRAGMA table_info(photos)")}
            fields = [
                name if name in columns else "NULL"
                for name in (
                    "absolute_path",
                    "embedding_source_sha256",
                    "metadata_json",
                    "phash",
                    "dhash",
                )
            ]
            if "absolute_path" not in columns:
                raise ValueError("Development database has no photo path inventory")
            for path, sha, meta, ph, dh in c.execute(
                "SELECT " + ",".join(fields) + " FROM photos"
            ):
                known_paths.add(str(Path(path).resolve()).casefold())
                for value in (sha, json.loads(meta or "{}").get("source_sha256")):
                    if value:
                        known_sha.add(value)
                if ph and dh:
                    known_hash.append((ph, dh))
    candidates, audit, seen = [], [], set(known_sha)
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not path.is_file():
            continue
        if str(path.resolve()).casefold() in known_paths:
            audit.append({"path": str(path), "excluded": "development path"})
            continue
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen:
            audit.append({"path": str(path), "excluded": "exact source hash"})
            continue
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            ph, dh = perceptual_hash(im), difference_hash(im)
        if any(
            hamming_distance(ph, a) <= 7 and hamming_distance(dh, b) <= 9
            for a, b in known_hash
        ):
            audit.append(
                {
                    "path": str(path),
                    "excluded": "conservative near-duplicate of development/earlier candidate",
                }
            )
            continue
        candidates.append(path)
        seen.add(sha)
        known_hash.append((ph, dh))
    if len(candidates) < 2 * args.pairs:
        raise ValueError(
            f"Only {len(candidates)} eligible photos for {args.pairs} pairs; do not reuse development photos"
        )
    random.Random(42).shuffle(candidates)
    # Freeze the relevant implementation fingerprint, not an invented release SHA.
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "ai/selection").glob("*.py")) + sorted(
        (root / "ai/aesthetics").glob("*.py")
    )
    method = hashlib.sha256(
        b"".join(x.name.encode() + x.read_bytes() for x in paths)
    ).hexdigest()
    manifest = dict(
        method_revision="source-sha256:" + method,
        label_provenance="human-pilot",
        development_and_memory_sha256=sorted(known_sha),
        development_and_memory_groups=["development"],
        cases=[
            dict(
                id=f"pair-{i + 1:02d}",
                group="single-local-collection-pilot",
                query="从你个人的审美出发，哪张照片更值得保留和分享？综合主体、构图、色彩，不必猜模型的答案。",
                left=str(candidates[2 * i]),
                right=str(candidates[2 * i + 1]),
            )
            for i in range(args.pairs)
        ],
    )
    args.output.mkdir(parents=True)
    (args.output / "source-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "sampling-audit.json").write_text(
        json.dumps(
            dict(
                eligible=len(candidates),
                excluded=audit,
                known_source_hashes=len(known_sha),
                limitation="Path/SHA/conservative perceptual hash exclusion only. Collection-level provenance unverified; one group; human pilot, NOT independent efficacy.",
                databases=[str(d.resolve()) for d in args.development_db],
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    build_bundle(manifest, args.output / "bundle")
    print(
        f"Prepared {args.pairs} unlabeled pairs; {len(audit)} exclusions. Human pilot only."
    )


if __name__ == "__main__":
    main()
