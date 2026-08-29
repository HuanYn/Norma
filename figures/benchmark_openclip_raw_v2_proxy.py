from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ai.index.embedding import create_embedding_provider
from ai.index.quality import analyze_quality
from ai.index.openclip_provider import _bridge_chinese_query


@dataclass(frozen=True, slots=True)
class QuerySpec:
    key: str
    category: str
    english: str
    chinese: str


QUERIES = (
    QuerySpec(
        key="architecture",
        category="travel architecture",
        english="travel architecture",
        chinese="旅行建筑",
    ),
    QuerySpec(
        key="city_night",
        category="city night photography",
        english="city night photography",
        chinese="城市夜景摄影",
    ),
    QuerySpec(
        key="mountain_landscape",
        category="mountain travel landscape",
        english="mountain travel landscape",
        chinese="山地旅行风景",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--album", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _metrics(ranked_relevance: list[int], positive_count: int) -> dict[str, float]:
    if positive_count <= 0:
        raise ValueError("each query needs at least one positive")
    first = next(
        (index for index, relevant in enumerate(ranked_relevance, start=1) if relevant),
        None,
    )

    def precision(cutoff: int) -> float:
        return sum(ranked_relevance[:cutoff]) / cutoff

    def recall(cutoff: int) -> float:
        return sum(ranked_relevance[:cutoff]) / positive_count

    def ndcg(cutoff: int) -> float:
        dcg = sum(
            relevant / math.log2(rank + 1)
            for rank, relevant in enumerate(ranked_relevance[:cutoff], start=1)
        )
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(positive_count, cutoff) + 1)
        )
        return dcg / ideal if ideal else 0.0

    return {
        "mrr": 0.0 if first is None else 1.0 / first,
        "precision_at_10": precision(10),
        "recall_at_20": recall(20),
        "ndcg_at_10": ndcg(10),
        "ndcg_at_20": ndcg(20),
    }


def _evaluate(
    *,
    method: str,
    query_language: str,
    specs: tuple[QuerySpec, ...],
    query_vectors: list[np.ndarray],
    image_vectors: list[np.ndarray],
    images: list[Path],
    labels: dict[str, str],
    quality_scores: dict[str, float],
    photo_ids: dict[str, str],
) -> dict[str, Any]:
    per_query: dict[str, Any] = {}
    for spec, query_vector in zip(specs, query_vectors, strict=True):
        ranked = sorted(
            (
                (round(float(np.dot(query_vector, vector)), 6), image.name)
                for image, vector in zip(images, image_vectors, strict=True)
            ),
            key=lambda item: (
                -item[0],
                -quality_scores[item[1]],
                photo_ids[item[1]],
            ),
        )
        relevance = [
            int(labels.get(filename) == spec.category) for _, filename in ranked
        ]
        positive_count = sum(
            1 for category in labels.values() if category == spec.category
        )
        per_query[spec.key] = {
            "category": spec.category,
            "query": spec.english if query_language == "english" else spec.chinese,
            "positive_count": positive_count,
            "metrics": _metrics(relevance, positive_count),
            "top_20": [
                {
                    "rank": rank,
                    "filename": filename,
                    "score": score,
                    "relevant": bool(relevant),
                }
                for rank, ((score, filename), relevant) in enumerate(
                    zip(ranked[:20], relevance[:20], strict=True), start=1
                )
            ],
        }
    metric_names = next(iter(per_query.values()))["metrics"]
    macro = {
        name: float(np.mean([item["metrics"][name] for item in per_query.values()]))
        for name in metric_names
    }
    return {
        "method": method,
        "query_language": query_language,
        "macro": macro,
        "per_query": per_query,
    }


def _validated(vectors: list[np.ndarray], dimension: int) -> dict[str, Any]:
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape[1:] != (dimension,):
        raise ValueError(f"unexpected vector matrix shape: {array.shape}")
    norms = np.linalg.norm(array, axis=1)
    return {
        "count": int(array.shape[0]),
        "dimension": dimension,
        "all_finite": bool(np.isfinite(array).all()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }


def main() -> None:
    args = _parse_args()
    album = args.album.resolve()
    attribution = json.loads((album / "ATTRIBUTION.json").read_text(encoding="utf-8"))
    labels = {item["file"]: item["search_term"] for item in attribution["images"]}
    images = sorted(
        path
        for path in album.iterdir()
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
    )
    album_id = uuid.uuid5(
        uuid.NAMESPACE_URL, f"norma:album:{str(album).casefold()}"
    ).hex
    quality_scores: dict[str, float] = {}
    photo_ids: dict[str, str] = {}
    for image_path in images:
        photo_ids[image_path.name] = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"norma:photo:{album_id}:{str(image_path.resolve()).casefold()}",
        ).hex
        with Image.open(image_path) as opened:
            quality_scores[image_path.name] = analyze_quality(
                ImageOps.exif_transpose(opened).convert("RGB")
            ).quality_score

    raw = create_embedding_provider(
        "openclip-multilingual",
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    lightweight = create_embedding_provider("lightweight")

    started = time.perf_counter()
    raw_images = raw.embed_images(images)
    raw_image_seconds = time.perf_counter() - started
    started = time.perf_counter()
    lightweight_images = lightweight.embed_images(images)
    lightweight_image_seconds = time.perf_counter() - started

    text_timings: dict[str, float] = {}

    def encode(provider: Any, key: str, texts: list[str]) -> list[np.ndarray]:
        started_text = time.perf_counter()
        vectors = [provider.embed_text(text) for text in texts]
        text_timings[key] = time.perf_counter() - started_text
        return vectors

    raw_english = encode(
        raw, "openclip_raw_english", [item.english for item in QUERIES]
    )
    raw_chinese = encode(
        raw, "openclip_raw_chinese", [item.chinese for item in QUERIES]
    )
    legacy_prompts = [_bridge_chinese_query(item.chinese) for item in QUERIES]
    legacy_chinese = encode(raw, "openclip_legacy_bridge_chinese", legacy_prompts)
    lightweight_english = encode(
        lightweight, "lightweight_english", [item.english for item in QUERIES]
    )
    lightweight_chinese = encode(
        lightweight, "lightweight_chinese", [item.chinese for item in QUERIES]
    )

    runs = [
        _evaluate(
            method="lightweight-semantic-v1",
            query_language="english",
            specs=QUERIES,
            query_vectors=lightweight_english,
            image_vectors=lightweight_images,
            images=images,
            labels=labels,
            quality_scores=quality_scores,
            photo_ids=photo_ids,
        ),
        _evaluate(
            method="openclip-xlm-roberta-base-vit-b-32-laion5b-raw-v2",
            query_language="english",
            specs=QUERIES,
            query_vectors=raw_english,
            image_vectors=raw_images,
            images=images,
            labels=labels,
            quality_scores=quality_scores,
            photo_ids=photo_ids,
        ),
        _evaluate(
            method="lightweight-semantic-v1",
            query_language="chinese",
            specs=QUERIES,
            query_vectors=lightweight_chinese,
            image_vectors=lightweight_images,
            images=images,
            labels=labels,
            quality_scores=quality_scores,
            photo_ids=photo_ids,
        ),
        _evaluate(
            method="openclip-xlm-roberta-base-vit-b-32-laion5b-zh-bridge-v1",
            query_language="chinese",
            specs=QUERIES,
            query_vectors=legacy_chinese,
            image_vectors=raw_images,
            images=images,
            labels=labels,
            quality_scores=quality_scores,
            photo_ids=photo_ids,
        ),
        _evaluate(
            method="openclip-xlm-roberta-base-vit-b-32-laion5b-raw-v2",
            query_language="chinese",
            specs=QUERIES,
            query_vectors=raw_chinese,
            image_vectors=raw_images,
            images=images,
            labels=labels,
            quality_scores=quality_scores,
            photo_ids=photo_ids,
        ),
    ]
    result = {
        "experiment_id": "openclip-raw-v2-wikimedia-proxy-20260828",
        "scope": {
            "candidate_images": len(images),
            "proxy_labeled_images": len(labels),
            "queries_per_language": len(QUERIES),
            "labels": "Wikimedia download search terms; not independent human relevance judgments",
            "cutoffs": [10, 20],
            "ranking_contract": "production score rounded to 6 decimals, then quality descending, then stable photo UUID",
        },
        "providers": {
            "openclip_raw": raw.name,
            "lightweight": lightweight.name,
            "legacy_image_vectors": "reused raw-v2 image vectors because the legacy subclass only changes text preparation",
        },
        "legacy_bridge_prompts": {
            item.chinese: prompt
            for item, prompt in zip(QUERIES, legacy_prompts, strict=True)
        },
        "timings_seconds": {
            "openclip_image_batch_total": raw_image_seconds,
            "openclip_image_mean": raw_image_seconds / len(images),
            "lightweight_image_total": lightweight_image_seconds,
            "lightweight_image_mean": lightweight_image_seconds / len(images),
            "text_groups": text_timings,
        },
        "validation": {
            "openclip_images": _validated(raw_images, raw.dimension),
            "lightweight_images": _validated(lightweight_images, lightweight.dimension),
            "raw_chinese_queries_preserved": [
                raw._prepare_text_query(item.chinese) == item.chinese
                for item in QUERIES
            ],
        },
        "runs": runs,
        "limitations": [
            "Only three query families are evaluated.",
            "Labels inherit Wikimedia search-category bias and are not blind human judgments.",
            "Nine synthetic derivative/duplicate candidates are unjudged and count as non-relevant.",
            "This pilot is directional integration evidence, not a general multilingual retrieval claim.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps({"output": str(args.output.resolve()), **result["scope"]}, indent=2)
    )


if __name__ == "__main__":
    main()
