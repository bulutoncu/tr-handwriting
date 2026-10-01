"""Controlled damage for the robustness experiment: how much can context recover
when parts of the handwriting are missing?"""

from __future__ import annotations

import random

import cv2
import numpy as np


def remove_small_marks(img: np.ndarray, max_size: float = 0.55) -> np.ndarray:
    """Erase small ink blobs (dots of i/ğ/ö/ü, detached cedillas, commas): every
    connected component whose bounding box is smaller than `max_size` x the
    median component height (roughly the letter x-height) in both directions. Imitates marks lost to a bad photo or a hasty writer.
    Marks that touch their letter (common in cursive) survive, as they would in reality."""
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    real = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= 4]  # ignore pixel noise
    if len(real) < 2:
        return img.copy()
    limit = max_size * float(np.median(stats[real, cv2.CC_STAT_HEIGHT]))
    small = [k for k in real
             if stats[k, cv2.CC_STAT_HEIGHT] < limit and stats[k, cv2.CC_STAT_WIDTH] < limit]
    out = img.copy()
    if not small:
        return out
    mask = np.isin(labels, small).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    out[mask] = int(np.percentile(img, 90))  # paint with paper colour
    return out


def erase_patches(img: np.ndarray, rng: random.Random, fraction: float = 0.1,
                  patch_width: int = 12) -> np.ndarray:
    """Paint over random vertical strips covering roughly `fraction` of the width."""
    out = img.copy()
    h, w = img.shape
    paper = int(np.percentile(img, 90))
    for _ in range(max(1, int(w * fraction / patch_width))):
        x = rng.randrange(0, max(1, w - patch_width))
        out[:, x:x + patch_width] = paper
    return out
