"""Verify or attempt to provision the historical preference-study fixture.

Wikimedia thumbnail URLs are mutable. Downloads therefore fail closed when an
upstream response no longer matches the historical byte pin; this command does
not silently replace the experiment input with a visually similar derivative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "fixtures" / "contextual_preference_wikimedia_20260814.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / ".norma" / "demo-album-eval"
MAX_FILE_BYTES = 30_000_000
MAX_TOTAL_BYTES = 1_000_000_000
USER_AGENT = "NormaPreferenceFixture/1.0 (https://github.com/HuanYn/Norma)"


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in fixture manifest: {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported contextual-preference fixture manifest")
    attribution = payload.get("attribution")
    images = payload.get("images")
    if not isinstance(attribution, dict) or not isinstance(images, list):
        raise ValueError("fixture manifest is missing attribution or images")
    if len(images) != 72:
        raise ValueError(
            "the pinned contextual-preference fixture must contain 72 images"
        )
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_record(record: Any, *, index: int) -> tuple[str, str, int, str]:
    if not isinstance(record, dict):
        raise ValueError(f"image record {index} must be an object")
    filename = record.get("file")
    url = record.get("download_url")
    expected_bytes = record.get("expected_byte_size")
    expected_sha256 = record.get("expected_sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or Path(filename).suffix.lower() not in {".jpg", ".jpeg"}
    ):
        raise ValueError(f"image record {index} has an unsafe filename")
    if not isinstance(url, str) or not url.startswith("https://upload.wikimedia.org/"):
        raise ValueError(f"image record {index} has an unsupported download URL")
    if (
        not isinstance(expected_bytes, int)
        or expected_bytes < 4
        or expected_bytes > MAX_FILE_BYTES
    ):
        raise ValueError(f"image record {index} has an invalid byte size")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(f"image record {index} has an invalid SHA-256")
    return filename, url, expected_bytes, expected_sha256


def _verify_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError(f"fixture path is not a file: {path.name}")
    if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha256:
        raise ValueError(f"existing fixture file failed its content pin: {path.name}")
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError(f"existing fixture file is not a JPEG: {path.name}")
    return True


def _download_file(
    *,
    target: Path,
    url: str,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
    digest = hashlib.sha256()
    received = 0
    first_bytes = b""
    try:
        with (
            urllib.request.urlopen(request, timeout=90) as response,
            partial.open("wb") as handle,
        ):
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_FILE_BYTES:
                raise ValueError(f"download exceeds the per-file limit: {target.name}")
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > MAX_FILE_BYTES or received > expected_bytes:
                    raise ValueError(
                        f"download exceeded its content pin: {target.name}"
                    )
                if not first_bytes:
                    first_bytes = chunk[:2]
                digest.update(chunk)
                handle.write(chunk)
        if (
            first_bytes != b"\xff\xd8"
            or received != expected_bytes
            or digest.hexdigest() != expected_sha256
        ):
            raise ValueError(
                "upstream response no longer matches the historical content pin: "
                f"{target.name}"
            )
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _attribution_bytes(manifest: dict[str, Any]) -> bytes:
    attribution = manifest["attribution"]
    images = manifest["images"]
    records = []
    for image in images:
        records.append(
            {
                "file": image["file"],
                "title": image["title"],
                "search_term": image["search_term"],
                "source_page": image["source_page"],
                "download_url": image["download_url"],
                "license": image["license"],
            }
        )
    payload = {
        "dataset": attribution["dataset"],
        "generated_at": attribution["generated_at"],
        "notice": attribution["notice"],
        "images": records,
    }
    # This ignored runtime file predates the repository's LF contract. Its byte
    # hash is part of the published experiment provenance, so reproduce the
    # original, explicitly declared CRLF/no-final-newline representation.
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return text.replace("\n", "\r\n").encode("utf-8")


def _write_attribution(output: Path, manifest: dict[str, Any]) -> None:
    encoded = _attribution_bytes(manifest)
    expected_sha256 = manifest["attribution"].get("sha256")
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("generated ATTRIBUTION.json failed its manifest hash")
    target = output / "ATTRIBUTION.json"
    if target.exists() and target.read_bytes() == encoded:
        return
    partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
    try:
        partial.write_bytes(encoded)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify or attempt to provision the historical 72-image Wikimedia "
            "fixture used by Norma's contextual-preference experiments; mutable "
            "upstream responses fail closed"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="verify the complete local fixture without making network requests",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest.resolve(strict=True))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, str, int, str]] = []
    filenames: set[str] = set()
    total_expected = 0
    for index, raw_record in enumerate(manifest["images"]):
        record = _validate_record(raw_record, index=index)
        if record[0] in filenames:
            raise ValueError(f"duplicate fixture filename: {record[0]}")
        filenames.add(record[0])
        total_expected += record[2]
        records.append(record)
    if total_expected > MAX_TOTAL_BYTES:
        raise ValueError("fixture exceeds the aggregate download limit")

    reused = 0
    downloaded = 0
    for filename, url, expected_bytes, expected_sha256 in records:
        target = output / filename
        if _verify_file(
            target,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        ):
            reused += 1
            continue
        if args.offline:
            raise FileNotFoundError(f"offline fixture file is missing: {filename}")
        _download_file(
            target=target,
            url=url,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        downloaded += 1

    _write_attribution(output, manifest)
    print(
        json.dumps(
            {
                "attribution_sha256": manifest["attribution"]["sha256"],
                "downloaded": downloaded,
                "file_count": len(records),
                "output": str(output),
                "reused": reused,
                "total_expected_bytes": total_expected,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
