"""Frozen OpenCLIP evidence and pairwise set diversity, never ground-truth labels.

The thresholds are explicit demo heuristics, not calibrated probabilities or an
accuracy claim. Unknown subjects cannot satisfy an exact category quota.
"""

from __future__ import annotations

import numpy as np

VERSION = "openclip-subject-and-pair-diversity-v1"
PROMPTS = {
    "portrait": (
        "A travel portrait of a person posing for the camera, the person is the main subject.",
        "A photograph of people, a close-up or full-body portrait outdoors.",
        "旅行人像照片，人物是画面主体，在风景前拍照。",
    ),
    "landscape": (
        "A scenic landscape photograph of mountains, lakes, forests or rivers, scenery is the main subject.",
        "A wide travel scenery photograph, natural landscape without a prominent person.",
        "自然风景摄影，雪山、湖泊、森林、河流或草原，景色是主体。",
    ),
    "other": (
        "A photograph of food, objects, documents or a room interior.",
        "A travel photograph of a building, architecture or a city street.",
        "食物、物品、文字、建筑或室内照片。",
    ),
}
MARGIN = 0.015
PAIR_THRESHOLD = 0.97
PENALTY_START = 0.80
PENALTY_WEIGHT = 0.35


def prototypes(provider):
    if not provider.model_backed:
        raise ValueError("类别判断需要预训练图文模型；不能使用规则向量替代。")
    result = {}
    for label, prompts in PROMPTS.items():
        vectors = np.asarray([provider.embed_text(p) for p in prompts])
        vector = vectors.mean(axis=0)
        result[label] = vector / np.linalg.norm(vector)
    return result


def classify(vector, text_vectors):
    scores = {k: float(np.dot(vector, v)) for k, v in text_vectors.items()}
    ordered = sorted(scores, key=lambda k: (-scores[k], k))
    margin = scores[ordered[0]] - scores[ordered[1]]
    label = ordered[0] if margin >= MARGIN else "uncertain"
    return {
        "label": label,
        "cosine_scores": scores,
        "margin": margin,
        "minimum_margin": MARGIN,
        "method": VERSION,
        "scope": "zero-shot subject estimate; not calibrated probability or identity recognition",
    }


def pair_constraints(vectors):
    """Pair exclusion (not transitive clusters) and graded redundancy cost.

    Two snow-mountain views are not duplicates merely for sharing a topic. Only
    very high visual similarity excludes a pair; weaker similarity has a soft
    cost that can be outweighed by relevance and quality.
    """
    conflicts, penalties = [], []
    matrix = np.asarray(vectors)
    if not len(matrix):
        return conflicts, penalties
    cosine = matrix @ matrix.T
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            similarity = float(cosine[left, right])
            if similarity >= PAIR_THRESHOLD:
                conflicts.append((left, right))
            elif similarity > PENALTY_START:
                penalties.append(
                    (
                        left,
                        right,
                        PENALTY_WEIGHT
                        * (similarity - PENALTY_START)
                        / (1 - PENALTY_START),
                    )
                )
    return conflicts, penalties
