"""Validate/export explicitly authorized proxy labels; never write user feedback.

No cloud requests, downloads, model training, or live database access. Image bytes
must match the pinned public source. Export is opt-in and refuses replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "fixtures" / "proxy_preferences_20261005.json"
DEFAULT_SOURCE = (
    PROJECT_ROOT / "fixtures" / "contextual_preference_wikimedia_20260814.json"
)
DEFAULT_ALBUM = PROJECT_ROOT / ".norma" / "demo-album-eval"
MAX_MANIFEST_BYTES = 2_000_000
MAX_IMAGE_BYTES = 30_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds size limit")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_object)
    if not isinstance(value, dict):
        raise ValueError("manifest must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _filename(value: Any) -> str:
    value = _text(value, "filename")
    if (
        Path(value).name != value
        or PureWindowsPath(value).name != value
        or ":" in value
        or value in {".", ".."}
    ):
        raise ValueError("unsafe filename")
    return value


def _hash(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("invalid SHA-256")
    return value


def _provenance(item: dict[str, Any]) -> None:
    if (
        item.get("source") != "assistant_proxy"
        or item.get("authorized_by_user") is not True
        or item.get("human_observed") is not False
    ):
        raise ValueError("proxy provenance/authorization must be preserved")


def validate_dataset(
    manifest: Path = DEFAULT_MANIFEST,
    *,
    source_manifest: Path = DEFAULT_SOURCE,
    public_album: Path | None = None,
) -> dict[str, Any]:
    """Validate provenance/schema/source pins; optionally verify local image bytes.

    Returning the dataset is not permission to treat its labels as human feedback.
    No user IDs, album IDs, learned feature vectors, or benchmark scores are made up.
    """
    dataset = _read_json(manifest)
    if type(dataset.get("schema_version")) is not int or dataset["schema_version"] != 1:
        raise ValueError("unsupported schema version")
    _provenance(dataset)
    _text(dataset.get("dataset_id"), "dataset_id")
    _text(dataset.get("authorization_scope"), "authorization_scope")
    _text(dataset.get("claim_boundary"), "claim_boundary")
    if dataset.get("usage") != "development_proxy_only":
        raise ValueError("proxy data must remain development-only")
    policy = dataset.get("evaluation", {})
    if (
        policy.get("eligible_as_ground_truth") is not False
        or policy.get("exclude_entire_source_corpus") is not True
    ):
        raise ValueError("held-out evaluation exclusion must be preserved")
    source_pin = dataset.get("public_source_manifest", {})
    if _filename(source_pin.get("file")) != source_manifest.name or _hash(
        source_pin.get("sha256")
    ) != sha256_file(source_manifest):
        raise ValueError("public source manifest content pin mismatch")
    source = _read_json(source_manifest)
    public = {entry["file"]: entry for entry in source["images"]}
    rubric_version = _text(dataset.get("rubric", {}).get("version"), "rubric version")
    assets = dataset.get("assets")
    pairs = dataset.get("pairs")
    if not isinstance(assets, list) or not 1 <= len(assets) <= 1000:
        raise ValueError("invalid asset list")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 1000:
        raise ValueError("invalid pair list")
    asset_by_name: dict[str, dict[str, Any]] = {}
    album_root = public_album.resolve(strict=True) if public_album is not None else None
    for asset in assets:
        name = _filename(asset.get("file"))
        if name in asset_by_name:
            raise ValueError("duplicate image asset")
        digest = _hash(asset.get("sha256"))
        size = asset.get("byte_size")
        if type(size) is not int or not 4 <= size <= MAX_IMAGE_BYTES:
            raise ValueError("invalid asset size")
        original = public.get(name)
        if original is None or (
            digest != original.get("expected_sha256")
            or size != original.get("expected_byte_size")
            or asset.get("source_page") != original.get("source_page")
            or asset.get("license_name")
            != original.get("license", {}).get("LicenseShortName")
            or asset.get("license_url") != original.get("license", {}).get("LicenseUrl")
            or asset.get("artist") != original.get("license", {}).get("Artist")
        ):
            raise ValueError(f"asset differs from public source record: {name}")
        _text(asset.get("inspection"), "inspection")
        if album_root is not None:
            path = (album_root / name).resolve(strict=True)
            if not path.is_relative_to(album_root) or not path.is_file():
                raise ValueError("asset escaped the public album")
            if path.stat().st_size != size or sha256_file(path) != digest:
                raise ValueError(f"image content pin mismatch: {name}")
        asset_by_name[name] = asset
    pair_ids: set[str] = set()
    for pair in pairs:
        _provenance(pair)
        pair_id = _text(pair.get("id"), "pair id")
        if pair_id in pair_ids:
            raise ValueError("duplicate pair id")
        pair_ids.add(pair_id)
        if pair.get("rubric_version") != rubric_version:
            raise ValueError("rubric version mismatch")
        for field in ("query_text", "rationale", "limitation"):
            _text(pair.get(field), field)
        left, right = pair.get("left_file"), pair.get("right_file")
        if left == right or left not in asset_by_name or right not in asset_by_name:
            raise ValueError("pair must reference two different pinned images")
        confidence = pair.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence must be finite in [0, 1]")
        choice = pair.get("choice")
        if choice == "preferred":
            if {pair.get("preferred_file"), pair.get("rejected_file")} != {left, right}:
                raise ValueError("preferred/rejected must match the displayed pair")
        elif choice in {"tie", "skip", "both_bad"}:
            if (
                pair.get("preferred_file") is not None
                or pair.get("rejected_file") is not None
            ):
                raise ValueError("non-ranking choices must not invent a winner")
        else:
            raise ValueError("unsupported pair choice")
    return dataset


def load_proxy_preferences(
    manifest: Path = DEFAULT_MANIFEST,
    *,
    allow_assistant_proxy: bool = False,
    source_manifest: Path = DEFAULT_SOURCE,
    public_album: Path = DEFAULT_ALBUM,
) -> list[dict[str, Any]]:
    """Return opt-in, byte-verified records with provenance and both image pins."""
    if allow_assistant_proxy is not True:
        raise ValueError("explicit allow_assistant_proxy=True is required")
    dataset = validate_dataset(
        manifest, source_manifest=source_manifest, public_album=public_album
    )
    assets = {asset["file"]: asset for asset in dataset["assets"]}
    return [
        {
            **pair,
            "dataset_id": dataset["dataset_id"],
            "dataset_sha256": sha256_file(manifest),
            "usage": dataset["usage"],
            "evaluation": dataset["evaluation"],
            "public_source_manifest": dataset["public_source_manifest"],
            "left_asset": assets[pair["left_file"]],
            "right_asset": assets[pair["right_file"]],
        }
        for pair in dataset["pairs"]
    ]


def assert_held_out_disjoint(
    evaluation_hashes: list[str],
    *,
    manifest: Path = DEFAULT_MANIFEST,
    source_manifest: Path = DEFAULT_SOURCE,
) -> None:
    """Guard for future evaluators, excluding the entire exposed source corpus."""
    validate_dataset(manifest, source_manifest=source_manifest)
    excluded = {row["expected_sha256"] for row in _read_json(source_manifest)["images"]}
    if excluded.intersection(_hash(item) for item in evaluation_hashes):
        raise ValueError("held-out data overlaps the proxy source corpus")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--public-album", type=Path, default=DEFAULT_ALBUM)
    parser.add_argument("--allow-assistant-proxy", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="opt-in JSONL export; refuses overwrite"
    )
    args = parser.parse_args(argv)
    if args.output is not None and not args.allow_assistant_proxy:
        parser.error("--output requires --allow-assistant-proxy")
    dataset = validate_dataset(
        args.manifest,
        source_manifest=args.source_manifest,
        public_album=args.public_album,
    )
    if args.output is not None:
        rows = load_proxy_preferences(
            args.manifest,
            allow_assistant_proxy=True,
            source_manifest=args.source_manifest,
            public_album=args.public_album,
        )
        # Exclusive create prevents accidental replacement of any existing file.
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
    print(
        json.dumps(
            {
                "dataset_id": dataset["dataset_id"],
                "dataset_sha256": sha256_file(args.manifest),
                "source": "assistant_proxy",
                "pairs": len(dataset["pairs"]),
                "images_verified": len(dataset["assets"]),
                "choice_counts": {
                    choice: sum(pair["choice"] == choice for pair in dataset["pairs"])
                    for choice in ("preferred", "tie", "skip", "both_bad")
                },
                "exported": args.output is not None,
                "live_feedback_written": False,
                "model_trained": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
