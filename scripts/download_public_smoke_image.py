"""Download Norma's licensed, content-pinned public multimodal smoke fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Gothic-architecture-banner.jpg"
DOWNLOAD_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/"
    "Gothic-architecture-banner.jpg/1920px-Gothic-architecture-banner.jpg"
)
EXPECTED_SHA256 = "5cd90ed5f970f80c7f39fdb5eca41c72bf21fd8dcdaf24ea287b4dc6b2405e9a"
EXPECTED_BYTES = 152_147
MAX_DOWNLOAD_BYTES = 5_000_000
USER_AGENT = "NormaPublicSmoke/1.0 (https://github.com/HuanYn/Norma)"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_existing(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return False
    if len(data) != EXPECTED_BYTES or _sha256(data) != EXPECTED_SHA256:
        raise ValueError(
            "the existing smoke image does not match the pinned public fixture"
        )
    return True


def _download() -> bytes:
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("the public smoke image exceeds the download limit")
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("the public smoke image exceeds the download limit")
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("the public smoke fixture is not a JPEG")
    if len(data) != EXPECTED_BYTES or _sha256(data) != EXPECTED_SHA256:
        raise ValueError("the downloaded public smoke image failed its content pin")
    return data


def _write_attribution(path: Path) -> None:
    record = {
        "file": path.name,
        "title": "Gothic-architecture-banner.jpg",
        "description": "Roof of Milan cathedral",
        "creator": "Traveler100",
        "source_page": SOURCE_PAGE,
        "download_url": DOWNLOAD_URL,
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
        "byte_size": EXPECTED_BYTES,
        "sha256": EXPECTED_SHA256,
    }
    attribution_path = path.with_suffix(".ATTRIBUTION.json")
    attribution_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify Norma's CC-licensed public smoke image"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".norma/public-smoke/gothic-architecture-banner.jpg"),
    )
    args = parser.parse_args()
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not _verified_existing(target):
        data = _download()
        partial = target.with_name(f"{target.name}.partial")
        try:
            partial.write_bytes(data)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
    _write_attribution(target)
    print(f"Verified public smoke image: {target}")
    print(f"SHA-256: {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
