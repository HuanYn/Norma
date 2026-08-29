from __future__ import annotations

import warnings
from io import BytesIO

from PIL import Image, UnidentifiedImageError


MAX_EVIDENCE_IMAGE_PIXELS = 64_000_000
MAX_EVIDENCE_TOTAL_PIXELS = 96_000_000


class EvidenceImagePixelLimitError(ValueError):
    """An evidence image would exceed the bounded local decode budget."""


class InvalidEvidenceImageError(ValueError):
    """Frozen evidence bytes are not a decodable image."""


def inspect_evidence_image_dimensions(content: bytes) -> tuple[int, int]:
    """Validate a frozen image header/stream and return bounded ``(width, height)``."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                width, height = image.size
                pixels = width * height
                if pixels < 1 or pixels > MAX_EVIDENCE_IMAGE_PIXELS:
                    raise EvidenceImagePixelLimitError(
                        "evidence image exceeds the 64 megapixel decode limit"
                    )
                image.verify()
    except EvidenceImagePixelLimitError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise EvidenceImagePixelLimitError(
            "evidence image exceeds the safe Pillow decompression limit"
        ) from error
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as error:
        raise InvalidEvidenceImageError("evidence image is not decodable") from error
    return width, height


def inspect_evidence_image(content: bytes) -> int:
    """Validate a frozen image header/stream and return its bounded pixel count."""

    width, height = inspect_evidence_image_dimensions(content)
    return width * height


def decode_evidence_image_rgb(content: bytes) -> Image.Image:
    """Decode one bounded evidence image after the service-side preflight."""

    inspect_evidence_image(content)
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as error:
        raise InvalidEvidenceImageError("evidence image is not decodable") from error


__all__ = [
    "EvidenceImagePixelLimitError",
    "InvalidEvidenceImageError",
    "MAX_EVIDENCE_IMAGE_PIXELS",
    "MAX_EVIDENCE_TOTAL_PIXELS",
    "decode_evidence_image_rgb",
    "inspect_evidence_image",
    "inspect_evidence_image_dimensions",
]
