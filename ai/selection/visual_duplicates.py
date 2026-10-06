"""Cached learned similarity with a spatial check for crop/zoom variants.

This folds a display view, never source files. Thresholds are explicit demo
heuristics, not a calibrated duplicate probability. No identity inference.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ai.index.embedding import embedding_cache_is_current
from ai.selection.service import _load_vector

VERSION = "openclip-spatial-curation-v1"
COSINE_DIRECT = 0.97
COSINE_GATE = 0.90
# This mode is similar-composition curation, not pixel duplicate detection.
# 35% reciprocal frame coverage allows roughly 1.7x crop/zoom variants; much
# wider environmental views must remain separate. Fixed development heuristics.
MIN_OVERLAP = 0.35
MIN_SPREAD = 0.08
MIN_CORRELATION = 0.75


def features(path):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("L")
        image.thumbnail((512, 512))
        gray = np.array(image)
    points, descriptors = cv2.SIFT_create(nfeatures=800).detectAndCompute(gray, None)
    return gray, np.array([p.pt for p in points], dtype=np.float32), descriptors


@lru_cache(maxsize=128)
def cached_features(source_key):
    # Caller verifies the original bytes before consulting this content-bound cache.
    return features(Path(source_key[0]))


@lru_cache(maxsize=4096)
def cached_spatial(left_key, right_key):
    return spatial_evidence(cached_features(left_key), cached_features(right_key))


def spatial_evidence(left, right):
    a, pa, da = left
    b, pb, db = right
    if da is None or db is None or min(len(da), len(db)) < 12:
        return None
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [
        m
        for pair in matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.7 * n.distance
    ]
    if len(good) < 12:
        return None
    src = np.float32([pa[m.queryIdx] for m in good])
    dst = np.float32([pb[m.trainIdx] for m in good])
    cv2.setRNGSeed(0)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if h is None or mask is None or not np.isfinite(h).all() or np.linalg.cond(h) > 1e8:
        return None
    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < 12 or float(inliers.mean()) < 0.6:
        return None

    def coverage(matrix, shape_a, shape_b):
        height, width = shape_a
        hb, wb = shape_b
        box = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        projected = cv2.perspectiveTransform(box[None], matrix)[0]
        if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
            return 0.0
        target = np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]])
        area, _ = cv2.intersectConvexConvex(projected, target)
        return float(area) / (wb * hb)

    overlap = min(
        coverage(h, a.shape, b.shape), coverage(np.linalg.inv(h), b.shape, a.shape)
    )
    spread = min(
        cv2.contourArea(cv2.convexHull(src[inliers])) / a.size,
        cv2.contourArea(cv2.convexHull(dst[inliers])) / b.size,
    )
    warped = cv2.warpPerspective(a, h, (b.shape[1], b.shape[0]))
    valid = cv2.warpPerspective(
        np.ones(a.shape, dtype=np.uint8),
        h,
        (b.shape[1], b.shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    if valid.sum() < 100 or np.std(warped[valid]) < 1 or np.std(b[valid]) < 1:
        return None
    correlation = float(np.corrcoef(warped[valid], b[valid])[0, 1])
    return {
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "overlap": overlap,
        "spread": spread,
        "correlation": correlation,
        "matched": overlap >= MIN_OVERLAP
        and spread >= MIN_SPREAD
        and correlation >= MIN_CORRELATION,
    }


def fold_similar(kept, rows, provider):
    if not provider.model_backed:
        raise ValueError("构图去重需要预训练图片特征，不能用规则向量冒充。")
    by_id = {r["id"]: r for r in rows}
    vectors, source_keys = {}, {}
    digest = hashlib.sha256()
    for photo in kept:
        row = by_id[photo["photo_id"]]
        if not embedding_cache_is_current(row, provider.name, strict_source_hash=True):
            raise ValueError("构图去重需要当前图片特征，请先完成语义模型准备。")
        vector = _load_vector(str(row["embedding_path"]), provider.dimension)
        vectors[photo["photo_id"]] = vector
        source_keys[photo["photo_id"]] = (
            str(row["absolute_path"]),
            row["embedding_source_sha256"],
            row["file_size"],
            row["source_mtime_ns"],
        )
        digest.update(photo["photo_id"].encode())
        digest.update(vector.astype("<f4").tobytes())
    retained, folded = [], []
    for photo in kept:  # Already ordered by learned aesthetic/technical score.
        matches = sorted(
            (
                (float(vectors[photo["photo_id"]] @ vectors[w["photo_id"]]), w)
                for w in retained
            ),
            key=lambda p: (-p[0], p[1]["photo_id"]),
        )
        for cosine, winner in matches:
            if cosine < COSINE_GATE:
                break
            geometry = None
            if cosine < COSINE_DIRECT:
                geometry = cached_spatial(
                    source_keys[photo["photo_id"]], source_keys[winner["photo_id"]]
                )
                if not geometry or not geometry["matched"]:
                    continue
            folded.append(
                photo
                | {
                    "reasons": [
                        f"画面高度相似，保留综合评分更高的 {winner['filename']}；可在折叠照片中查看原图"
                    ],
                    "visual_duplicate_of": winner["photo_id"],
                    "visual_similarity": {
                        "cosine": cosine,
                        "spatial": geometry,
                        "method": VERSION,
                    },
                }
            )
            break
        else:
            retained.append(photo)
            continue
        if cosine < COSINE_GATE:
            retained.append(photo)
    return (
        retained,
        folded,
        {
            "version": VERSION,
            "provider": provider.name,
            "embedding_digest": digest.hexdigest(),
            "direct_cosine": COSINE_DIRECT,
            "spatial_gate_cosine": COSINE_GATE,
            "min_overlap": MIN_OVERLAP,
            "min_spread": MIN_SPREAD,
            "min_correlation": MIN_CORRELATION,
            "opencv_version": cv2.__version__,
            "assignments": [[p["photo_id"], p["visual_duplicate_of"]] for p in folded],
            "folded_count": len(folded),
            "transitive_merging": False,
        },
    )
