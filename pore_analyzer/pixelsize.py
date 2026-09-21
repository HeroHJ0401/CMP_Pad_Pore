"""
Work out µm/px for an image without making the user do arithmetic.

Two routes:
  read_pixel_size(path)  - pull it out of the file's own metadata, which SEM
                           vendors write into TIFFs (Zeiss, FEI/Thermo, ImageJ,
                           or a plain TIFF resolution tag).
  from_scale_bar(...)    - the universal fallback: the length of the scale bar
                           in pixels plus the value printed next to it. Works on
                           screenshots and re-saved JPEGs, which carry no
                           metadata at all.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from PIL import Image

__all__ = ["read_pixel_size", "from_scale_bar", "UNITS_UM", "format_pixel_size",
           "ScaleStore"]

# multiplier that converts a length in that unit into µm
UNITS_UM = {
    "pm": 1e-6,
    "nm": 1e-3,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,          # U+03BC, which some vendors use instead of U+00B5
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
}


def format_pixel_size(um_per_px: float) -> str:
    if um_per_px >= 1:
        return f"{um_per_px:.4g} µm/px"
    return f"{um_per_px:.4g} µm/px  ({um_per_px * 1000:.4g} nm/px)"


# --------------------------------------------------------------------------
# Scale bar
# --------------------------------------------------------------------------

def from_scale_bar(length_px: float, length_value: float, unit: str) -> float:
    """µm per pixel from a bar `length_px` long labelled `length_value` `unit`."""
    if length_px <= 0:
        raise ValueError("스케일바 길이가 0입니다. 바의 양 끝을 다시 지정하십시오.")
    if length_value <= 0:
        raise ValueError("스케일바에 적힌 길이는 0보다 커야 합니다.")
    key = unit.strip()
    if key not in UNITS_UM:
        raise ValueError(f"알 수 없는 단위입니다: {unit}")
    return (length_value * UNITS_UM[key]) / float(length_px)


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

def _texts_from_tiff(img) -> list:
    """Every tag value that might carry text, as strings."""
    out = []
    tags = getattr(img, "tag_v2", None) or getattr(img, "tag", None)
    if not tags:
        return out
    for tag_id in (34118, 34682, 270, 271, 272, 305, 33560, 50431):
        try:
            v = tags.get(tag_id)
        except Exception:
            v = None
        if v is None:
            continue
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        if isinstance(v, (tuple, list)):
            v = " ".join(str(x) for x in v)
        out.append(str(v))
    return out


def _from_text(text: str) -> Optional[Tuple[float, str]]:
    """Look for a vendor's pixel-size statement in a blob of tag text."""
    # FEI / Thermo Fisher: an INI block with metres
    m = re.search(r"PixelWidth\s*=\s*([0-9.eE+-]+)", text)
    if m:
        try:
            metres = float(m.group(1))
            if 0 < metres < 1:
                return metres * 1e6, "FEI/Thermo 메타데이터"
        except ValueError:
            pass

    # Zeiss: "Image Pixel Size = 4.04 nm", with or without the AP_ key name
    m = re.search(r"(?:Image\s*Pixel\s*Size|AP_IMAGE_PIXEL_SIZE)\D{0,20}?"
                  r"([0-9]+\.?[0-9]*)\s*(pm|nm|µm|μm|um|mm)",
                  text, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1)) * UNITS_UM[m.group(2).lower().replace("μ", "µ")
                                               if m.group(2).lower() != "um" else "um"]
            if val > 0:
                return val, "Zeiss 메타데이터"
        except (ValueError, KeyError):
            pass

    # Hitachi sidecar style: "PixelSize 4.04" with a separate unit line
    m = re.search(r"PixelSize\s*[=:]?\s*([0-9]+\.?[0-9]*)\s*(pm|nm|µm|μm|um|mm)?",
                  text, re.IGNORECASE)
    if m:
        try:
            unit = (m.group(2) or "nm").lower().replace("μm", "µm")
            unit = "um" if unit == "um" else unit
            val = float(m.group(1)) * UNITS_UM.get(unit, 1e-3)
            if val > 0:
                return val, "Hitachi 메타데이터"
        except ValueError:
            pass
    return None


def _from_resolution_tag(img) -> Optional[Tuple[float, str]]:
    """Plain TIFF XResolution, including the way ImageJ writes microns."""
    tags = getattr(img, "tag_v2", None)
    if not tags:
        return None
    try:
        xres = tags.get(282)
        unit = tags.get(296, 2)          # 1 = none, 2 = inch, 3 = cm
    except Exception:
        return None
    if not xres:
        return None
    try:
        xres = float(xres[0]) / float(xres[1]) if isinstance(xres, tuple) else float(xres)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if xres <= 0:
        return None

    desc = ""
    try:
        d = tags.get(270)
        desc = d.decode("utf-8", "replace") if isinstance(d, bytes) else str(d or "")
    except Exception:
        pass

    m = re.search(r"unit\s*=\s*(\w+)", desc, re.IGNORECASE)
    if m:                                 # ImageJ: resolution is per `unit`
        u = m.group(1).lower()
        u = {"micron": "um", "microns": "um", "micrometer": "um"}.get(u, u)
        if u in UNITS_UM:
            return UNITS_UM[u] / xres, "ImageJ 메타데이터"

    if unit == 3:                         # pixels per cm
        return 1e4 / xres, "TIFF 해상도 태그"
    if unit == 2:                         # pixels per inch
        return 25400.0 / xres, "TIFF 해상도 태그"
    return None


def read_pixel_size(path: str) -> Optional[Tuple[float, str]]:
    """
    Best-effort µm/px from the file itself.

    Returns (um_per_px, where_it_came_from) or None. A value is only returned
    when it is physically plausible for an SEM image (1 pm to 1 mm per pixel),
    because a wrong pixel size silently rescales every area in the report.
    """
    try:
        with Image.open(path) as img:
            found = None
            for text in _texts_from_tiff(img):
                found = _from_text(text)
                if found:
                    break
            if not found:
                found = _from_resolution_tag(img)

            # A sidecar text file is how some Hitachi and JEOL exports ship it
            if not found:
                stem = os.path.splitext(path)[0]
                for ext in (".txt", ".TXT"):
                    side = stem + ext
                    if os.path.isfile(side):
                        try:
                            with open(side, "r", encoding="utf-8",
                                      errors="replace") as fh:
                                found = _from_text(fh.read(20000))
                        except OSError:
                            pass
                        if found:
                            found = (found[0], found[1] + " (사이드카 .txt)")
                            break
    except Exception:
        return None

    if not found:
        return None
    value, source = found
    if not (1e-6 < value < 1e3):          # 1 pm .. 1 mm per pixel
        return None
    return value, source


# --------------------------------------------------------------------------
# Remembering scales between runs
# --------------------------------------------------------------------------

class ScaleStore:
    """
    Remember a measured µm/px per image file across runs.

    Measuring a scale bar is manual work, and a batch can hold dozens of
    images. Losing that on exit would mean redoing it every session, so the
    values are kept in a small JSON file keyed by absolute path. Values read
    from metadata are not stored: those are free to recompute and would only
    go stale if the file changed.
    """

    MAX_ENTRIES = 4000

    def __init__(self, path: str = None):
        self.path = path or self._default_path()
        self.data = {}
        self.load()

    @staticmethod
    def _default_path() -> str:
        base = (os.environ.get("APPDATA")
                or os.environ.get("XDG_CONFIG_HOME")
                or os.path.join(os.path.expanduser("~"), ".config"))
        return os.path.join(base, "cmp-pore-analyzer", "scales.json")

    def load(self):
        import json
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                self.data = {k: v for k, v in loaded.items()
                             if isinstance(v, dict) and
                             isinstance(v.get("um_per_px"), (int, float)) and
                             0 < v["um_per_px"] < 1e3}
        except (OSError, ValueError):
            self.data = {}       # a missing or broken store is not an error

    def save(self):
        import json
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            items = list(self.data.items())[-self.MAX_ENTRIES:]
            self.data = dict(items)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)     # never leave a half-written store
            return True
        except OSError:
            return False                   # remembering is best-effort

    def get(self, image_path: str):
        """(um_per_px, source) remembered for this file, or None."""
        rec = self.data.get(os.path.abspath(image_path))
        if not rec:
            return None
        return rec["um_per_px"], rec.get("source", "이전 측정값")

    def put(self, image_path: str, um_per_px: float, source: str):
        if not (um_per_px and 0 < um_per_px < 1e3):
            return
        self.data[os.path.abspath(image_path)] = {
            "um_per_px": float(um_per_px), "source": source}

    def forget(self, image_path: str):
        self.data.pop(os.path.abspath(image_path), None)
