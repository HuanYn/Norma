from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = np.asarray(
        image.convert("L").resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    low = cv2.dct(gray)[:hash_size, :hash_size]
    median = float(np.median(low))
    return _bits_to_hex((low > median).ravel())


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = np.asarray(
        image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    return _bits_to_hex((gray[:, 1:] > gray[:, :-1]).ravel())


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def assign_similarity_groups(hashes: list[tuple[str, str]]) -> dict[int, str]:
    """Group conservative near-duplicates; singletons have no group."""

    parents = list(range(len(hashes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            phash_distance = hamming_distance(hashes[left][0], hashes[right][0])
            dhash_distance = hamming_distance(hashes[left][1], hashes[right][1])
            if phash_distance <= 7 and dhash_distance <= 9:
                union(left, right)

    members: dict[int, list[int]] = {}
    for index in range(len(hashes)):
        members.setdefault(find(index), []).append(index)

    result: dict[int, str] = {}
    group_number = 0
    for indices in sorted(members.values(), key=lambda values: values[0]):
        if len(indices) < 2:
            continue
        group_number += 1
        group_id = f"sim_{group_number:03d}"
        for index in indices:
            result[index] = group_id
    return result


def _bits_to_hex(bits: np.ndarray) -> str:
    binary = "".join("1" if bool(bit) else "0" for bit in bits)
    return f"{int(binary, 2):0{len(binary) // 4}x}"

