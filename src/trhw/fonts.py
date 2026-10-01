"""Handwriting-style fonts from the Google Fonts repository (OFL / Apache licensed).

Fonts with full Turkish coverage render real Turkish text. Fonts missing
ç/ğ/ı/İ/ş are still useful: the recognizer only reads base shapes, so they
render the base-shape version of the line instead.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont

from .text import TURKISH_SPECIAL

_BASE = "https://raw.githubusercontent.com/google/fonts/main/"
FONT_PATHS = [
    "ofl/caveat/Caveat%5Bwght%5D.ttf", "ofl/caveatbrush/CaveatBrush-Regular.ttf",
    "ofl/kalam/Kalam-Regular.ttf", "ofl/patrickhand/PatrickHand-Regular.ttf",
    "ofl/dancingscript/DancingScript%5Bwght%5D.ttf", "ofl/indieflower/IndieFlower-Regular.ttf",
    "ofl/shadowsintolighttwo/ShadowsIntoLightTwo-Regular.ttf",
    "ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf",
    "ofl/annieuseyourtelescope/AnnieUseYourTelescope-Regular.ttf",
    "ofl/cedarvillecursive/Cedarville-Cursive.ttf", "ofl/gloriahallelujah/GloriaHallelujah.ttf",
    "ofl/grapenuts/GrapeNuts-Regular.ttf", "ofl/itim/Itim-Regular.ttf", "ofl/mali/Mali-Regular.ttf",
    "ofl/marckscript/MarckScript-Regular.ttf", "ofl/overtherainbow/OvertheRainbow.ttf",
    "ofl/pangolin/Pangolin-Regular.ttf", "ofl/playpensans/PlaypenSans%5Bwght%5D.ttf",
    "ofl/sedgwickave/SedgwickAve-Regular.ttf", "ofl/sriracha/Sriracha-Regular.ttf",
    "ofl/zeyada/Zeyada.ttf",
    # partial Turkish coverage -> used for base-shape rendering only
    # (all-caps fonts like Rock Salt are excluded: they would make the labels wrong)
    "ofl/gochihand/GochiHand-Regular.ttf", "ofl/handlee/Handlee-Regular.ttf",
    "apache/homemadeapple/HomemadeApple-Regular.ttf", "ofl/kristi/Kristi-Regular.ttf",
    "ofl/reeniebeanie/ReenieBeanie.ttf",
    "apache/schoolbell/Schoolbell-Regular.ttf", "ofl/nothingyoucoulddo/NothingYouCouldDo.ttf",
    "ofl/coveredbyyourgrace/CoveredByYourGrace.ttf", "ofl/neucha/Neucha.ttf",
]
REQUIRED_BASE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?'\"()-/%&"


def font_charset(path: str | Path) -> frozenset[str]:
    return frozenset(chr(cp) for cp in TTFont(str(path)).getBestCmap())


def download_fonts(out_dir: str | Path) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rel in FONT_PATHS:
        name = rel.rsplit("/", 1)[-1].replace("%5B", "[").replace("%5D", "]")
        target = out_dir / name
        if not target.exists():
            try:
                urllib.request.urlretrieve(_BASE + rel, target)
            except Exception as exc:  # noqa: BLE001 - one missing font is not fatal
                print(f"skip {name}: {exc}")
                continue
        chars = font_charset(target)
        if not set(REQUIRED_BASE) <= chars:
            print(f"skip {name}: missing base glyphs")
            continue
        manifest.append({"file": name, "turkish": set(TURKISH_SPECIAL) <= chars})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(font_dir: str | Path) -> list[dict]:
    font_dir = Path(font_dir)
    entries = json.loads((font_dir / "manifest.json").read_text())
    for e in entries:
        e["path"] = str(font_dir / e["file"])
    return entries
