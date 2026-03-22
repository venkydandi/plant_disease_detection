from __future__ import annotations

import numpy as np
from PIL import Image

IMG_SIZE = (128, 128)


def normalize_image(image: Image.Image, image_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Resize and scale image to [0, 1] as float32."""
    image = image.convert("RGB").resize(image_size)
    return np.asarray(image, dtype=np.float32) / 255.0


def extract_features(img_array: np.ndarray) -> np.ndarray:
    """Build compact color-texture feature vector for classical ML."""
    flat = img_array.reshape(-1, 3)

    means = np.mean(flat, axis=0)
    stds = np.std(flat, axis=0)
    mins = np.min(flat, axis=0)
    maxs = np.max(flat, axis=0)

    # 16-bin histogram per channel captures leaf color distribution.
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(flat[:, channel], bins=16, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float32))

    hist_features = np.concatenate(hist_parts)
    features = np.concatenate([means, stds, mins, maxs, hist_features]).astype(np.float32)
    return features
