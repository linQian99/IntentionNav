"""Preserve a detector crop's entire extent before CLIP's center crop."""
from __future__ import annotations

from PIL import Image


def square_pad_crop(crop: Image.Image | None) -> Image.Image | None:
    """Center the unchanged crop in a square with CLIP-mean neutral padding.

    No extra scene pixels or target labels are introduced. The image processor
    can then resize and center-crop without discarding the long-axis endpoints.
    """
    if crop is None or crop.width == crop.height:
        return crop
    size = max(crop.size)
    padded = Image.new("RGB", (size, size), (123, 117, 104))
    padded.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return padded
