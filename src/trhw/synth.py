"""Render text lines in handwriting fonts, with augmentations that imitate a
phone photo of a notebook: wobbly baseline, uneven spacing, slant, rotation,
stroke thickness, blur, uneven paper and noise.

Every sample carries two labels:
  text - the real Turkish line
  base - the base-shape line the recognizer is trained to output
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .text import to_base


@dataclass
class SynthConfig:
    out_height: int = 64
    font_size: tuple[int, int] = (38, 58)
    word_jitter: float = 0.08      # vertical jitter per word, fraction of font size
    space_scale: tuple[float, float] = (0.8, 1.6)
    shear: tuple[float, float] = (-0.35, 0.25)
    max_rotation_deg: float = 2.0
    p_thicken: float = 0.3
    p_thin: float = 0.2
    p_blur: float = 0.4
    noise_std: tuple[float, float] = (0.0, 12.0)
    p_ruled_line: float = 0.3


def _render_words(text: str, font: ImageFont.FreeTypeFont, rng: random.Random,
                  cfg: SynthConfig, size: int) -> np.ndarray:
    words = text.split(" ")
    space = font.getlength(" ")
    widths = [font.getlength(w) for w in words]
    pad = size
    width = int(sum(widths) + space * cfg.space_scale[1] * len(words) + 2 * pad)
    height = int(size * 2.2)
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    x = pad
    baseline = height * 0.3
    drift = rng.uniform(-0.03, 0.03) * size  # slow baseline drift across the line
    for i, (w, wd) in enumerate(zip(words, widths)):
        y = baseline + drift * i + rng.gauss(0, cfg.word_jitter * size)
        draw.text((x, y), w, font=font, fill=255)
        x += wd + space * rng.uniform(*cfg.space_scale)
    ink = np.array(img)
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return ink
    m = size // 6
    return ink[max(0, ys.min() - m):ys.max() + m, max(0, xs.min() - m):xs.max() + m]


def _geometric(ink: np.ndarray, rng: random.Random, cfg: SynthConfig) -> np.ndarray:
    h, w = ink.shape
    shear = rng.uniform(*cfg.shear)
    extra = int(abs(shear) * h)
    matrix = np.float32([[1, shear, extra if shear < 0 else 0], [0, 1, 0]])
    ink = cv2.warpAffine(ink, matrix, (w + extra, h))
    angle = rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg)
    h, w = ink.shape
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    new_h = int(h + abs(np.sin(np.radians(angle))) * w) + 2
    rot[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(ink, rot, (w, new_h))


def _stroke(ink: np.ndarray, rng: random.Random, cfg: SynthConfig) -> np.ndarray:
    kernel = np.ones((2, 2), np.uint8)
    r = rng.random()
    if r < cfg.p_thicken:
        return cv2.dilate(ink, kernel, iterations=1)
    if r < cfg.p_thicken + cfg.p_thin:
        return cv2.erode(ink, kernel, iterations=1)
    return ink


def _paper(ink: np.ndarray, rng: random.Random, cfg: SynthConfig) -> np.ndarray:
    h, w = ink.shape
    np_rng = np.random.default_rng(rng.randrange(2**32))
    # low-frequency lighting variation
    small = np_rng.normal(0, 1, (4, max(2, w // 64)))
    light = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    paper = rng.uniform(200, 250) + light * rng.uniform(0, 12)
    if rng.random() < cfg.p_ruled_line:
        y = int(h * rng.uniform(0.75, 0.95))
        paper[max(0, y - 1):y + 1, :] -= rng.uniform(20, 60)
    ink_level = rng.uniform(10, 90)
    alpha = ink.astype(np.float32) / 255.0
    img = paper * (1 - alpha) + ink_level * alpha
    img += np_rng.normal(0, rng.uniform(*cfg.noise_std), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def render_line(text: str, font_path: str, rng: random.Random,
                cfg: SynthConfig | None = None) -> np.ndarray:
    """Return a grayscale uint8 image (dark ink on light paper) of fixed height."""
    cfg = cfg or SynthConfig()
    size = rng.randint(*cfg.font_size)
    font = ImageFont.truetype(font_path, size)
    ink = _render_words(text, font, rng, cfg, size)
    ink = _geometric(ink, rng, cfg)
    ink = _stroke(ink, rng, cfg)
    img = _paper(ink, rng, cfg)
    if rng.random() < cfg.p_blur:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.2))
    h, w = img.shape
    new_w = max(1, int(w * cfg.out_height / h))
    return cv2.resize(img, (new_w, cfg.out_height), interpolation=cv2.INTER_AREA)


def make_sample(text: str, fonts: list[dict], rng: random.Random,
                cfg: SynthConfig | None = None) -> tuple[np.ndarray, dict]:
    """Pick a font; if it lacks Turkish glyphs, render the base-shape text instead."""
    font = rng.choice(fonts)
    shown = text if font["turkish"] else to_base(text)
    img = render_line(shown, font["path"], rng, cfg)
    return img, {"text": text, "base": to_base(text), "font": font["file"],
                 "marks_visible": shown == text}
