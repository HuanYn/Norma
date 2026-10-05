from __future__ import annotations

import base64
import binascii
import io
import warnings
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_BASE64_LENGTH = ((MAX_IMAGE_BYTES + 2) // 3) * 4


class VideoOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="blur, distorted face, deformed hands, flicker, text, watermark", max_length=2000)
    width: int = Field(default=640, ge=256, le=1280)
    height: int = Field(default=384, ge=256, le=1280)
    num_frames: int = Field(default=49, ge=9, le=121)
    num_inference_steps: int = Field(default=20, ge=4, le=50)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=8.0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @field_validator("prompt", "negative_prompt")
    @classmethod
    def clean_prompt(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\t\r" for char in value):
            raise ValueError("Prompt contains control characters")
        return value.strip()

    @model_validator(mode="after")
    def validate_geometry(self) -> "VideoOptions":
        if not self.prompt:
            raise ValueError("Prompt must not be blank")
        if self.width % 32 or self.height % 32:
            raise ValueError("Wan TI2V dimensions must be divisible by 32")
        if self.width * self.height > 1280 * 704:
            raise ValueError("Maximum output area is 1280 x 704 pixels")
        if self.num_frames % 4 != 1:
            raise ValueError("num_frames must equal 4k + 1")
        return self


class VideoSubmit(VideoOptions):
    image_base64: str = Field(min_length=4, max_length=MAX_BASE64_LENGTH, repr=False)
    upload_confirmed: Literal[True]

    @field_validator("upload_confirmed", mode="before")
    @classmethod
    def explicit_confirmation(cls, value):
        if value is not True:
            raise ValueError("Explicit upload confirmation is required")
        return value


def decode_image(value: str) -> Image.Image:
    """Bounded JPEG/PNG only; return pixels with no uploaded metadata attached."""
    if len(value) > MAX_BASE64_LENGTH:
        raise ValueError("Image exceeds 8 MiB")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds 8 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG are supported")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds 20 megapixels")
                if min(source.size) < 32:
                    raise ValueError("Image is too small")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported")
                source.load()
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                clean = Image.new("RGB", oriented.size)
                clean.paste(oriented)
                return clean
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError("Image is corrupt or too large") from exc
