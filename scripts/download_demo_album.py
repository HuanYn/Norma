from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "NormaPortfolioDemo/0.1 (https://github.com/HuanYn/Norma)"
SEARCHES = (
    "travel architecture",
    "city night photography",
    "mountain travel landscape",
    "street photography travel",
    "cafe travel photography",
    "historic building travel",
    "coast sunset travel",
    "urban portrait travel",
)
LICENSE_KEYS = (
    "LicenseShortName",
    "LicenseUrl",
    "Artist",
    "Credit",
    "Attribution",
    "AttributionRequired",
    "UsageTerms",
    "ImageDescription",
)


def api(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode({"format": "json", "formatversion": "2", **params})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def discover(count: int) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    per_query = max(12, (count // len(SEARCHES)) * 3)
    for term in SEARCHES:
        payload = api(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": term,
                "gsrnamespace": "6",
                "gsrlimit": str(min(50, per_query)),
                "prop": "imageinfo",
                "iiprop": "url|mime|size|sha1",
                "iiurlwidth": "1600",
            }
        )
        for page in payload.get("query", {}).get("pages", []):
            title = page.get("title", "")
            info = (page.get("imageinfo") or [{}])[0]
            if not title.casefold().endswith((".jpg", ".jpeg")):
                continue
            if info.get("mime") != "image/jpeg" or not info.get("thumburl"):
                continue
            candidates.setdefault(title, {"title": title, **info, "search_term": term})
        if len(candidates) >= count * 2:
            break
        time.sleep(0.2)
    return list(candidates.values())[:count]


def add_license_metadata(items: list[dict[str, Any]]) -> None:
    for start in range(0, len(items), 10):
        batch = items[start : start + 10]
        payload = api(
            {
                "action": "query",
                "titles": "|".join(item["title"] for item in batch),
                "prop": "imageinfo",
                "iiprop": "extmetadata",
                "iiextmetadatafilter": "|".join(LICENSE_KEYS),
                "iiextmetadatalanguage": "en",
            }
        )
        by_title = {page["title"]: page for page in payload.get("query", {}).get("pages", [])}
        for item in batch:
            page = by_title.get(item["title"], {})
            metadata = ((page.get("imageinfo") or [{}])[0]).get("extmetadata", {})
            item["license"] = {
                key: _plain(metadata.get(key, {}).get("value", "")) for key in LICENSE_KEYS
            }
            item["source_page"] = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(
                item["title"].replace(" ", "_"), safe="/:"
            )
        time.sleep(0.2)


def download(items: list[dict[str, Any]], destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        filename = f"{index:03d}_{item['sha1'][:12]}.jpg"
        target = destination / filename
        request = urllib.request.Request(item["thumburl"], headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if not data.startswith(b"\xff\xd8"):
                raise ValueError("response was not a JPEG")
            target.write_bytes(data)
        except Exception as error:
            print(f"skip {item['title']}: {error}", flush=True)
            continue
        manifest.append(
            {
                "file": filename,
                "title": item["title"],
                "search_term": item["search_term"],
                "source_page": item["source_page"],
                "download_url": item["thumburl"],
                "license": item["license"],
            }
        )
        print(f"[{len(manifest):03d}/{len(items):03d}] {filename}", flush=True)
    return manifest


def _plain(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(value))).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a licensed Wikimedia Commons demo album")
    parser.add_argument("--count", type=int, default=72)
    parser.add_argument("--output", type=Path, default=Path(".norma/demo-album"))
    args = parser.parse_args()
    if not 1 <= args.count <= 200:
        parser.error("--count must be between 1 and 200")

    items = discover(args.count)
    add_license_metadata(items)
    manifest = download(items, args.output)
    manifest_path = args.output / "ATTRIBUTION.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "Norma Wikimedia Commons demo album",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "notice": "Verify each source page before external redistribution; license terms vary per file.",
                "images": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Downloaded {len(manifest)} JPEGs to {args.output.resolve()}", flush=True)
    print(f"Attribution manifest: {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
