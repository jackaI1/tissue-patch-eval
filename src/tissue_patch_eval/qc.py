"""Fixed, label-free screening heuristics. Flags are review cues, not diagnoses."""

import numpy as np
from scipy import ndimage

THRESHOLDS = {
    "laplacian_variance_below": 80.0,
    "background_fraction_above": 0.80,
    "pen_fraction_above": 0.005,
    "pen_component_pixels_at_least": 12,
}
FLAGS = ("blur", "pen", "background")


def validate_image(image):
    if image.shape != (96, 96, 3) or image.dtype != np.uint8:
        raise ValueError("expected 96 x 96 x 3 uint8 RGB image")


def quality_flags(image: np.ndarray) -> dict:
    validate_image(image)
    rgb = image.astype(np.float32)
    r, g, b = np.moveaxis(rgb, -1, 0)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    saturation = (rgb.max(2) - rgb.min(2)) / np.maximum(rgb.max(2), 1)
    background_fraction = float(((rgb.min(2) > 220) & (saturation < 0.12)).mean())
    laplacian_variance = float(ndimage.laplace(gray).var())
    # Saturated blue, green, or red marks; purple nuclei alone should not qualify.
    mask = (
        ((b > 130) & (b > 1.5 * r) & (b > 1.2 * g) & (r < 100))
        | ((g > 130) & (g > 1.4 * r) & (g > 1.3 * b))
        | ((r > 180) & (g < 80) & (b < 100))
    )
    components, _ = ndimage.label(mask)
    sizes = np.bincount(components.ravel())
    keep = sizes >= THRESHOLDS["pen_component_pixels_at_least"]
    keep[0] = False
    pen_fraction = float(keep[components].mean())
    return {
        "blur": laplacian_variance < THRESHOLDS["laplacian_variance_below"],
        "pen": pen_fraction > THRESHOLDS["pen_fraction_above"],
        "background": background_fraction > THRESHOLDS["background_fraction_above"],
        "laplacian_variance": laplacian_variance,
        "background_fraction": background_fraction,
        "pen_fraction": pen_fraction,
    }


def extract_features(image: np.ndarray) -> np.ndarray:
    """Color distribution + texture in full patch and labelled central region only."""
    validate_image(image)
    features = []
    for region in (image, image[32:64, 32:64]):
        rgb = region.astype(np.float32) / 255
        pixels = rgb.reshape(-1, 3)
        features.extend(pixels.mean(0))
        features.extend(pixels.std(0))
        features.extend(np.quantile(pixels, [0.1, 0.5, 0.9], axis=0).ravel())
        for channel in range(3):
            features.extend(np.histogram(rgb[:, :, channel], bins=8, range=(0, 1))[0] / len(pixels))
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        dx = ndimage.sobel(gray, axis=0)
        dy = ndimage.sobel(gray, axis=1)
        features.extend(
            [np.hypot(dx, dy).mean(), np.hypot(dx, dy).std(), ndimage.laplace(gray).var()]
        )
    return np.asarray(features, dtype=np.float32)
