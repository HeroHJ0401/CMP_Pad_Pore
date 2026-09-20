"""
CMP pad SEM image -> open-pore fraction (개공률) analysis core.

Pipeline (paper-equivalent):
    grayscale -> normalize -> Gaussian smoothing (sigma px)
    -> fixed-threshold binarization (dark = candidate opening)
    -> morphological opening (disk r)
    -> connected-component labeling
    -> exclude objects with equivalent diameter < d_min and frame-touching objects
    -> shape metrics: circularity = 4*pi*A / P**2, solidity = A / A_convexhull
    -> open-pore fraction = sum(area of objects with solidity >= S_cut) / field area

No GUI dependencies in this module so it can be unit-tested / run headless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, filters
from PIL import Image

__all__ = ["Params", "Result", "analyze_path", "analyze_array", "robustness_sweep"]


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass
class Params:
    """Analysis parameters. Defaults reproduce the published analysis."""

    pixel_size_um: float = 0.404      # µm per pixel
    sigma_px: float = 1.2             # Gaussian smoothing sigma, in pixels
    threshold: float = 0.45           # normalized brightness threshold (dark < thr)
    opening_radius_px: int = 2        # morphological opening disk radius
    min_diam_um: float = 2.0          # drop objects below this equivalent diameter
    solidity_cut: float = 0.90        # convex-opening criterion
    exclude_border: bool = True       # drop frame-touching objects
    crop_bottom_px: int = 0           # crop SEM info bar (rows removed from bottom)
    normalize: str = "fixed"          # "fixed" (v/255) | "minmax" | "otsu"

    def copy_with(self, **kw) -> "Params":
        d = asdict(self)
        d.update(kw)
        return Params(**d)


@dataclass
class Result:
    """Per-image analysis output."""

    source: str = ""
    width_px: int = 0
    height_px: int = 0
    field_area_um2: float = 0.0
    field_w_um: float = 0.0
    field_h_um: float = 0.0

    n_objects: int = 0                # objects surviving all filters
    n_rejected_small: int = 0
    n_rejected_border: int = 0

    dark_area_fraction: float = 0.0   # raw dark pixel fraction (diagnostic only)
    open_pore_fraction: float = 0.0   # PRIMARY metric: solidity>=cut area / field area
    border_area_fraction: float = 0.0 # area lost to frame-touching objects

    circularity_median: float = float("nan")
    circularity_q1: float = float("nan")
    circularity_q3: float = float("nan")
    solidity_median: float = float("nan")
    solidity_q1: float = float("nan")
    solidity_q3: float = float("nan")
    eqdiam_median_um: float = float("nan")
    object_density_per_mm2: float = 0.0
    concave_ratio: float = float("nan")   # fraction of objects with solidity < cut
    mean_convex_deficiency: float = float("nan")

    otsu_threshold: float = float("nan")  # reference only

    # heavy payloads (not written to CSV)
    objects: list = field(default_factory=list, repr=False)
    overlay: Optional[np.ndarray] = field(default=None, repr=False)
    binary: Optional[np.ndarray] = field(default=None, repr=False)

    def summary_row(self) -> dict:
        """Flat dict for CSV / table display."""
        return {
            "file": self.source,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "field_w_um": round(self.field_w_um, 2),
            "field_h_um": round(self.field_h_um, 2),
            "n_objects": self.n_objects,
            "object_density_per_mm2": round(self.object_density_per_mm2, 1),
            "open_pore_fraction_pct": round(100 * self.open_pore_fraction, 2),
            "dark_area_fraction_pct": round(100 * self.dark_area_fraction, 2),
            "circularity_median": _r(self.circularity_median, 4),
            "circularity_IQR": f"{_r(self.circularity_q1,3)}-{_r(self.circularity_q3,3)}",
            "solidity_median": _r(self.solidity_median, 4),
            "solidity_IQR": f"{_r(self.solidity_q1,3)}-{_r(self.solidity_q3,3)}",
            "eqdiam_median_um": _r(self.eqdiam_median_um, 3),
            "concave_ratio_pct": _r(100 * self.concave_ratio, 1),
            "mean_convex_deficiency_pct": _r(100 * self.mean_convex_deficiency, 1),
            "n_rejected_small": self.n_rejected_small,
            "n_rejected_border": self.n_rejected_border,
            "border_area_fraction_pct": round(100 * self.border_area_fraction, 2),
            "otsu_threshold_ref": _r(self.otsu_threshold, 3),
        }


def _r(x, n):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        return round(float(x), n)
    except Exception:
        return ""


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------

def load_gray(path: str, crop_bottom_px: int = 0) -> np.ndarray:
    """Load an image as float grayscale in [0, 1]."""
    img = Image.open(path)
    if img.mode not in ("L", "I;16", "I"):
        img = img.convert("L")
    arr = np.asarray(img).astype(np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.max() > 255:                 # 16-bit
        arr = arr / 65535.0
    else:
        arr = arr / 255.0
    if crop_bottom_px > 0 and crop_bottom_px < arr.shape[0]:
        arr = arr[: arr.shape[0] - crop_bottom_px, :]
    return np.clip(arr, 0.0, 1.0)


def detect_info_bar(gray: np.ndarray) -> int:
    """
    Detect a solid SEM data/annotation bar at the bottom of the frame.

    Deliberately conservative: a false positive silently removes real image
    area from the denominator, which is worse than missing a bar the user can
    crop manually. A row only counts as banner if it is essentially flat
    (std < 0.01) AND close to pure black or pure white, and the whole run must
    be at least 1.5 % of the image height while staying under one third of it.
    Returns the number of rows to remove from the bottom, or 0.
    """
    h = gray.shape[0]
    row_std = gray.std(axis=1)
    row_mean = gray.mean(axis=1)
    body_mean = float(np.median(row_mean[: max(1, h // 2)]))

    n = 0
    limit = h // 3
    for i in range(h - 1, h - 1 - limit, -1):
        flat = row_std[i] < 0.01
        extreme = (row_mean[i] < 0.12) or (row_mean[i] > 0.88)
        if flat and extreme:
            n += 1
        else:
            break

    if n < max(6, int(h * 0.015)):
        return 0

    # Extend upward through annotation rows inside the same band: a row of
    # white text on a black bar is not flat, but almost every pixel in it is
    # still at one extreme of the range.
    dark_bar = float(np.mean(row_mean[h - n:])) < 0.5
    for i in range(h - 1 - n, h - 1 - limit, -1):
        row = gray[i]
        frac = float(np.mean(row < 0.12) if dark_bar else np.mean(row > 0.88))
        if frac >= 0.75:
            n += 1
        else:
            break

    # the band must also be clearly different from the image body
    if abs(float(np.mean(row_mean[h - n:])) - body_mean) < 0.15:
        return 0
    return n


# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------

def analyze_array(gray: np.ndarray, p: Params, source: str = "") -> Result:
    """Run the full pipeline on a float grayscale image in [0, 1]."""
    res = Result(source=source)
    h, w = gray.shape
    res.height_px, res.width_px = h, w
    res.field_w_um = w * p.pixel_size_um
    res.field_h_um = h * p.pixel_size_um
    res.field_area_um2 = res.field_w_um * res.field_h_um

    # ---- normalization -------------------------------------------------
    g = gray
    if p.normalize == "minmax":
        lo, hi = np.percentile(g, [0.5, 99.5])
        if hi > lo:
            g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)

    # ---- smoothing -----------------------------------------------------
    sm = ndi.gaussian_filter(g, sigma=p.sigma_px) if p.sigma_px > 0 else g

    # ---- threshold -----------------------------------------------------
    try:
        res.otsu_threshold = float(filters.threshold_otsu(sm))
    except Exception:
        res.otsu_threshold = float("nan")

    thr = res.otsu_threshold if p.normalize == "otsu" else p.threshold
    binary = sm < thr                     # dark = candidate opening
    res.dark_area_fraction = float(binary.mean())

    # ---- morphological opening ----------------------------------------
    if p.opening_radius_px and p.opening_radius_px > 0:
        binary = morphology.opening(binary, morphology.disk(p.opening_radius_px))

    # ---- labeling & metrics -------------------------------------------
    lbl = measure.label(binary, connectivity=2)
    props = measure.regionprops(lbl)

    px_area = p.pixel_size_um ** 2
    min_area_px = math.pi * (p.min_diam_um / p.pixel_size_um / 2.0) ** 2

    kept, border_area_px = [], 0.0
    for pr in props:
        if pr.area < min_area_px:
            res.n_rejected_small += 1
            continue
        r0, c0, r1, c1 = pr.bbox
        touches = (r0 == 0) or (c0 == 0) or (r1 == h) or (c1 == w)
        if touches:
            res.n_rejected_border += 1
            border_area_px += pr.area
            if p.exclude_border:
                continue
        try:
            eqd_px = pr.equivalent_diameter_area   # scikit-image >= 0.24
        except AttributeError:
            eqd_px = pr.equivalent_diameter
        perim = pr.perimeter
        circ = (4.0 * math.pi * pr.area / (perim ** 2)) if perim > 0 else float("nan")
        try:
            sol = float(pr.solidity)
        except Exception:
            sol = float("nan")
        kept.append({
            "label": int(pr.label),
            "area_um2": pr.area * px_area,
            "area_px": int(pr.area),
            "eqdiam_um": eqd_px * p.pixel_size_um,
            "circularity": min(circ, 1.0) if np.isfinite(circ) else float("nan"),
            "circularity_raw": circ,
            "solidity": sol,
            "centroid_rc": (float(pr.centroid[0]), float(pr.centroid[1])),
            "touches_border": bool(touches),
        })

    res.objects = kept
    res.n_objects = len(kept)
    res.border_area_fraction = border_area_px / float(h * w)

    if kept:
        circ = np.array([o["circularity"] for o in kept], dtype=float)
        sol = np.array([o["solidity"] for o in kept], dtype=float)
        eqd = np.array([o["eqdiam_um"] for o in kept], dtype=float)
        area = np.array([o["area_um2"] for o in kept], dtype=float)

        res.circularity_median = float(np.nanmedian(circ))
        res.circularity_q1, res.circularity_q3 = [float(x) for x in np.nanpercentile(circ, [25, 75])]
        res.solidity_median = float(np.nanmedian(sol))
        res.solidity_q1, res.solidity_q3 = [float(x) for x in np.nanpercentile(sol, [25, 75])]
        res.eqdiam_median_um = float(np.nanmedian(eqd))
        res.object_density_per_mm2 = len(kept) / (res.field_area_um2 / 1.0e6)

        convex_mask = sol >= p.solidity_cut
        res.open_pore_fraction = float(area[convex_mask].sum() / res.field_area_um2)
        res.concave_ratio = float((~convex_mask).sum() / len(kept))
        res.mean_convex_deficiency = float(np.nanmean(1.0 - sol))
    else:
        res.open_pore_fraction = 0.0

    res.binary = binary
    res.overlay = _make_overlay(gray, lbl, kept, p)
    return res


def _make_overlay(gray: np.ndarray, lbl: np.ndarray, kept: list, p: Params) -> np.ndarray:
    """RGB uint8 overlay: convex openings green, rejected/concave red."""
    base = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    rgb = np.dstack([base, base, base]).astype(np.float64)

    keep_labels = {o["label"]: o["solidity"] for o in kept}
    if not keep_labels:
        return rgb.astype(np.uint8)

    lut_convex = np.zeros(lbl.max() + 1, dtype=bool)
    lut_concave = np.zeros(lbl.max() + 1, dtype=bool)
    for lab, sol in keep_labels.items():
        if np.isfinite(sol) and sol >= p.solidity_cut:
            lut_convex[lab] = True
        else:
            lut_concave[lab] = True

    mconv = lut_convex[lbl]
    mconc = lut_concave[lbl]

    # fill, then draw boundaries brighter
    rgb[mconv] = 0.55 * rgb[mconv] + 0.45 * np.array([46, 204, 113])
    rgb[mconc] = 0.55 * rgb[mconc] + 0.45 * np.array([231, 76, 60])

    edge_c = mconv ^ ndi.binary_erosion(mconv)
    edge_x = mconc ^ ndi.binary_erosion(mconc)
    rgb[edge_c] = np.array([26, 188, 156])
    rgb[edge_x] = np.array([192, 57, 43])

    return np.clip(rgb, 0, 255).astype(np.uint8)


def analyze_path(path: str, p: Params) -> Result:
    """Load an image file and analyze it."""
    gray = load_gray(path, crop_bottom_px=p.crop_bottom_px)
    import os
    return analyze_array(gray, p, source=os.path.basename(path))


# --------------------------------------------------------------------------
# Robustness sweep
# --------------------------------------------------------------------------

def robustness_sweep(gray: np.ndarray, p: Params,
                     thresholds=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
                     solidities=(0.85, 0.90, 0.95)) -> list:
    """
    Recompute the open-pore fraction across a threshold x solidity grid.
    Absolute values move a lot; the point is to show the ordering is stable
    when two conditions are compared under identical settings.
    """
    rows = []
    for t in thresholds:
        pt = p.copy_with(threshold=t)
        r = analyze_array(gray, pt)
        for s in solidities:
            area = sum(o["area_um2"] for o in r.objects
                       if np.isfinite(o["solidity"]) and o["solidity"] >= s)
            rows.append({
                "threshold": t,
                "solidity_cut": s,
                "open_pore_fraction_pct": round(100 * area / r.field_area_um2, 2),
                "n_objects": r.n_objects,
            })
    return rows


# --------------------------------------------------------------------------
# Bootstrap comparison of two images
# --------------------------------------------------------------------------

def bootstrap_median_diff(a: list, b: list, n_boot: int = 4000, seed: int = 0):
    """
    95% CI for the median difference (a - b) of a per-object metric.
    NOTE: objects within one field of view are spatially correlated, so this
    CI describes stability within the field, not between-field uncertainty.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (np.median(rng.choice(a, a.size, replace=True))
                    - np.median(rng.choice(b, b.size, replace=True)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "diff": float(np.median(a) - np.median(b)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_a": int(a.size),
        "n_b": int(b.size),
    }
