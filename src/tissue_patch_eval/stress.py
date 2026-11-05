"""Controlled damage checks, kept separate from naturally occurring QC associations."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def damage(image, kind):
    result = Image.fromarray(image)
    if kind == "blur":
        result = result.filter(ImageFilter.GaussianBlur(3))
    elif kind == "pen":
        ImageDraw.Draw(result).line((5, 10, 88, 80), fill=(0, 40, 255), width=8)
    elif kind == "background":
        ImageDraw.Draw(result).rectangle((0, 0, 87, 95), fill="white")
    else:
        raise ValueError("unknown damage type")
    return np.asarray(result)
