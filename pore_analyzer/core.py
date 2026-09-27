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

__all__ = ["Params", "Result", "analyze_path", "analyze_array",
           "robustness_sweep", "detect_bands", "load_gray",
           "prepare", "pooled_otsu", "unique_labels",
           "normalization_window", "contrast_report", "otsu_separability"]


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
    crop_top_px: int = 0              # crop letterboxing (rows removed from top)

    # How the dark/bright decision is made. These two are independent:
    #   normalize_contrast  rescales each image onto its own intensity span,
    #                       cancelling gain/offset differences between images.
    #   threshold_mode      "fixed"      -> use `threshold` as given
    #                       "otsu"       -> pick it per image
    #                       "otsu_batch" -> pick ONE from all images together
    normalize_contrast: bool = False
    threshold_mode: str = "fixed"
    norm_low_pct: float = 1.0         # percentile mapped to 0 when normalizing
    norm_high_pct: float = 99.0       # percentile mapped to 1 when normalizing

    # Otsu returns a threshold whatever it is given, including a field with no
    # pores in it. `separability` (Otsu's eta) says whether two brightness
    # classes were really there; below this the reading is flagged rather than
    # reported as if it were sound. 0.70 sits in the gap measured on synthetic
    # images (0.90-0.94 genuine / 0.44-0.64 degenerate) and is deliberately on
    # the permissive side, since only synthetic evidence stands behind it.
    min_separability: float = 0.70

    # Smoothing and opening are pixel operations, so at two magnifications the
    # same pixel count is a different physical size. Setting these in µm keeps
    # the physical scale identical across images taken at different mags, which
    # is what makes their numbers comparable. None = use the _px value above.
    sigma_um: Optional[float] = None
    opening_radius_um: Optional[float] = None

    def resolved_sigma_px(self) -> float:
        if self.sigma_um is None:
            return self.sigma_px
        return self.sigma_um / self.pixel_size_um if self.pixel_size_um > 0 else 0.0

    def resolved_opening_px(self) -> int:
        if self.opening_radius_um is None:
            return int(self.opening_radius_px)
        if self.pixel_size_um <= 0:
            return 0
        return max(0, int(round(self.opening_radius_um / self.pixel_size_um)))

    def copy_with(self, **kw) -> "Params":
        d = asdict(self)
        d.update(kw)
        return Params(**d)


@dataclass
class Result:
    """Per-image analysis output."""

    source: str = ""            # label shown to the user (unique within a batch)
    source_path: str = ""       # the file it came from
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
    invalid_pore_fraction: float = 0.0  # solidity<cut  area / field area
    total_pore_fraction: float = 0.0    # every kept object's area / field area
    valid_count_ratio: float = float("nan")   # 1 - concave_ratio, by count
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

    otsu_threshold: float = float("nan")    # this image's own Otsu value
    effective_threshold: float = float("nan")  # what was actually applied
    threshold_source: str = ""              # 고정 / Otsu(이미지별) / Otsu(일괄)
    separability: float = float("nan")      # Otsu eta on the prepared image
    dynamic_range: float = float("nan")     # width of the normalization window
    gray_levels: int = 0                    # 8-bit levels inside that window
    norm_low_pct_used: float = float("nan")
    pixel_size_um: float = float("nan")     # the scale used for THIS image
    sigma_px_used: float = float("nan")
    opening_px_used: int = 0
    warnings: list = field(default_factory=list)

    # heavy payloads (not written to CSV)
    objects: list = field(default_factory=list, repr=False)
    overlay: Optional[np.ndarray] = field(default=None, repr=False)
    binary: Optional[np.ndarray] = field(default=None, repr=False)

    def summary_row(self) -> dict:
        """Flat dict for CSV / table display."""
        return {
            "file": self.source,
            "path": self.source_path,
            "pixel_size_um": _r(self.pixel_size_um, 6),
            "width_px": self.width_px,
            "height_px": self.height_px,
            "field_w_um": round(self.field_w_um, 2),
            "field_h_um": round(self.field_h_um, 2),
            "n_objects": self.n_objects,
            "object_density_per_mm2": round(self.object_density_per_mm2, 1),
            # area fractions: total = valid + invalid
            "pore_fraction_pct": round(100 * self.total_pore_fraction, 2),
            "valid_pore_fraction_pct": round(100 * self.open_pore_fraction, 2),
            "invalid_pore_fraction_pct": round(100 * self.invalid_pore_fraction, 2),
            # bright (non-pore) share of the field; the dark share is kept too
            "non_pore_area_pct": round(100 * (1.0 - self.dark_area_fraction), 2),
            "dark_area_fraction_pct": round(100 * self.dark_area_fraction, 2),
            "circularity_median": _r(self.circularity_median, 4),
            "circularity_IQR": f"{_r(self.circularity_q1,3)}-{_r(self.circularity_q3,3)}",
            "solidity_median": _r(self.solidity_median, 4),
            "solidity_IQR": f"{_r(self.solidity_q1,3)}-{_r(self.solidity_q3,3)}",
            "eqdiam_median_um": _r(self.eqdiam_median_um, 3),
            "valid_ratio_pct": _r(100 * self.valid_count_ratio, 1),
            "concave_ratio_pct": _r(100 * self.concave_ratio, 1),
            "mean_convex_deficiency_pct": _r(100 * self.mean_convex_deficiency, 1),
            "n_rejected_small": self.n_rejected_small,
            "n_rejected_border": self.n_rejected_border,
            "border_area_fraction_pct": round(100 * self.border_area_fraction, 2),
            "effective_threshold": _r(self.effective_threshold, 4),
            "threshold_source": self.threshold_source,
            "otsu_threshold_ref": _r(self.otsu_threshold, 3),
            "separability": _r(self.separability, 3),
            "dynamic_range": _r(self.dynamic_range, 3),
            "gray_levels": self.gray_levels,
            "warning": " / ".join(self.warnings),
            "sigma_px_used": _r(self.sigma_px_used, 3),
            "sigma_um_used": _r(self.sigma_px_used * self.pixel_size_um, 4),
            "opening_px_used": self.opening_px_used,
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

def load_gray(path: str, crop_bottom_px: int = 0, crop_top_px: int = 0) -> np.ndarray:
    """Load an image as float grayscale in [0, 1], optionally trimming bands."""
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

    h = arr.shape[0]
    top = max(0, int(crop_top_px))
    bot = max(0, int(crop_bottom_px))
    if top + bot < h:
        arr = arr[top: h - bot if bot else h, :]
    return np.clip(arr, 0.0, 1.0)


def _scan_band(gray: np.ndarray, from_bottom: bool) -> int:
    """
    Count rows belonging to a solid band at one edge of the frame.

    Deliberately conservative: a false positive silently removes real image
    area from the denominator, which is worse than missing a band the user can
    crop manually. A row only counts as band if it is essentially flat
    (std < 0.01) AND close to pure black or pure white, and the whole run must
    be at least 1.5 % of the image height while staying under one third of it.
    """
    h = gray.shape[0]
    row_std = gray.std(axis=1)
    row_mean = gray.mean(axis=1)
    limit = h // 3

    # index of the k-th row inward from the chosen edge
    def idx(k):
        return h - 1 - k if from_bottom else k

    body = row_mean[h // 4: 3 * h // 4]
    body_mean = float(np.median(body)) if body.size else float(np.median(row_mean))

    n = 0
    while n < limit:
        i = idx(n)
        flat = row_std[i] < 0.01
        extreme = (row_mean[i] < 0.12) or (row_mean[i] > 0.88)
        if flat and extreme:
            n += 1
        else:
            break

    if n < max(6, int(h * 0.015)):
        return 0

    band_rows = [idx(k) for k in range(n)]
    dark_band = float(np.mean(row_mean[band_rows])) < 0.5

    # Extend inward through annotation rows inside the same band: a row of
    # white text on a black bar is not flat, but almost every pixel in it is
    # still at one extreme of the range.
    while n < limit:
        row = gray[idx(n)]
        frac = float(np.mean(row < 0.12) if dark_band else np.mean(row > 0.88))
        if frac >= 0.75:
            n += 1
        else:
            break

    band_rows = [idx(k) for k in range(n)]
    if abs(float(np.mean(row_mean[band_rows])) - body_mean) < 0.15:
        return 0
    return n


def detect_bands(gray: np.ndarray) -> tuple:
    """
    Detect solid bands at the top and bottom of the frame and return
    (top_rows, bottom_rows). Covers both the SEM data bar, which sits at the
    bottom, and the black letterboxing a screen capture adds at both edges.
    Either value is 0 when no band is found. The two runs never overlap.
    """
    h = gray.shape[0]
    bottom = _scan_band(gray, from_bottom=True)
    top = _scan_band(gray, from_bottom=False)
    if top + bottom >= h:               # degenerate (near-uniform image)
        return 0, 0
    return top, bottom


def detect_info_bar(gray: np.ndarray) -> int:
    """Bottom band only. Kept for callers that predate detect_bands()."""
    return detect_bands(gray)[1]


# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------

def _minority_fraction(gray: np.ndarray) -> float:
    """
    Rough share of the image taken by the darker of its two brightness classes,
    from a plain Otsu split of the raw pixels. Used only to keep the
    normalization anchors out of the majority population — never as a result.
    """
    try:
        t = float(filters.threshold_otsu(gray))
    except Exception:
        return float("nan")
    return float((gray < t).mean())


def normalization_window(gray: np.ndarray, p: Params) -> tuple:
    """
    The two grey levels that normalization maps to 0 and 1, plus the
    percentiles actually used: (lo, hi, lo_pct, hi_pct).

    The anchors must sit OUTSIDE the two populations, not inside one of them.
    With the default lower anchor at 1 % and a field whose pores cover only
    0.5 %, the 1st percentile lands in the bright matrix: the stretch then
    spreads matrix noise across the whole range and clips every pore together
    at 0. Otsu duly splits the noise and the reading came out at 5.16 % against
    a true 0.41 % — an order of magnitude, silently.

    So each anchor is held below half the population on its own side, estimated
    from a plain Otsu split of the raw image. For an ordinary field (pores ~9 %)
    the estimate is far above 1 % and nothing moves; it only bites when the
    anchor would otherwise have fallen inside a population.
    """
    lo_pct = max(0.0, float(p.norm_low_pct))
    hi_pct = min(100.0, float(p.norm_high_pct))
    dark = _minority_fraction(gray)
    if np.isfinite(dark) and 0.0 < dark < 1.0:
        lo_pct = min(lo_pct, 100.0 * dark / 2.0)
        hi_pct = max(hi_pct, 100.0 - 100.0 * (1.0 - dark) / 2.0)
    lo, hi = np.percentile(gray, [lo_pct, hi_pct])
    return float(lo), float(hi), lo_pct, hi_pct


def prepare(gray: np.ndarray, p: Params) -> np.ndarray:
    """Normalize (optionally) and smooth — everything before the threshold."""
    g = gray
    if p.normalize_contrast:
        lo, hi, _, _ = normalization_window(g, p)
        if hi > lo:
            g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    sigma = p.resolved_sigma_px()
    return ndi.gaussian_filter(g, sigma=sigma) if sigma > 0 else g


def contrast_report(gray: np.ndarray, p: Params) -> dict:
    """
    How much brightness information this image actually carries, measured on
    the raw pixels before normalization.

      dynamic_range    hi - lo of the normalization window, in 0-1 grey units.
      gray_levels      distinct 8-bit levels inside that window. This is the
                       one that sets a floor on what any threshold can resolve:
                       a window 5 levels wide cannot be split finely, however
                       far it is stretched afterwards.
      norm_low_pct     the lower anchor actually used (see normalization_window).
    """
    lo, hi, lo_pct, hi_pct = normalization_window(gray, p)
    inside = gray[(gray >= lo) & (gray <= hi)]
    levels = int(np.unique(np.round(inside * 255.0)).size) if inside.size else 0
    return {
        "dynamic_range": float(hi - lo),
        "gray_levels": levels,
        "norm_low_pct": float(lo_pct),
        "norm_high_pct": float(hi_pct),
    }


def otsu_separability(sm: np.ndarray, nbins: int = 1024) -> float:
    """
    Otsu's own separability measure eta = between-class variance at the chosen
    threshold / total variance, on the already-prepared image. 1 means two
    perfectly separated brightness classes, 0 means one indivisible blob.

    This is the honest answer to "is a threshold meaningful on this image at
    all". Otsu always returns a threshold, even for a field with no pores in
    it: on synthetic no-pore images it split the noise and reported ~5 % open
    pore area out of nothing. eta is what tells those apart — measured 0.90-0.94
    for genuine pores at any contrast (0.939 at full contrast, 0.938 after
    compressing it 16x, so it does not merely restate contrast), against 0.64
    for a pore-free field and 0.44 where normalization had collapsed.
    """
    sm = np.asarray(sm).ravel()
    if sm.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(sm, bins=edges)
    pr = counts.astype(np.float64)
    total = pr.sum()
    if total <= 0:
        return float("nan")
    pr /= total
    mu = float((pr * centers).sum())
    var_total = float((pr * (centers - mu) ** 2).sum())
    if var_total <= 0:
        return 0.0
    w1 = np.cumsum(pr)
    w2 = np.cumsum(pr[::-1])[::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.cumsum(pr * centers) / w1
        m2 = (np.cumsum((pr * centers)[::-1]) / w2[::-1])[::-1]
        vb = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    vb = np.nan_to_num(vb, nan=-1.0, posinf=-1.0, neginf=-1.0)
    best = float(vb.max())
    if best <= 0:
        return 0.0
    return float(min(1.0, best / var_total))


POOLED_NBINS = 4096         # bin resolution of the pooled histogram: 1/4096


def _otsu_from_hist(counts: np.ndarray, centers: np.ndarray) -> float:
    """
    Otsu's threshold from a histogram, using scikit-image's own convention:
    class 0 is bins 0..i, class 1 is bins i+1.., and the returned value is
    centers[argmax]. Implemented here rather than calling
    filters.threshold_otsu(hist=...) so the result cannot drift with the
    scikit-image version bundled into a frozen build.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.sum() <= 0 or counts.size < 2:
        return float("nan")
    w1 = np.cumsum(counts)
    w2 = np.cumsum(counts[::-1])[::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.cumsum(counts * centers) / w1
        m2 = (np.cumsum((counts * centers)[::-1]) / w2[::-1])[::-1]
        var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    var = np.nan_to_num(var, nan=-1.0, posinf=-1.0, neginf=-1.0)
    if not np.any(var > 0):
        return float("nan")
    return float(centers[int(np.argmax(var))])


def pooled_otsu(grays, p: Params, nbins: int = POOLED_NBINS) -> float:
    """
    One Otsu threshold computed from several images at once.

    Per-image Otsu cancels brightness differences but also moves with the real
    thing being measured: an image with genuinely more open pore area gets a
    different threshold, so part of the difference under test is absorbed. A
    threshold pooled over the images being compared adapts to the set's actual
    greyscale range while staying identical for every image in it.

    Every pixel of every image is counted, through an accumulated histogram on
    a FIXED grid over [0, 1]. Two properties follow, and both are the point:

      * The result does not depend on the order the images arrive in. The
        earlier version drew a random subsample per image from one shared
        Generator, so the draw for image k depended on how many images came
        before it — reordering the same files moved the threshold, and with it
        the reported numbers. Measured on synthetic sets: 0.5 % of the value
        when the histogram was cleanly bimodal, 5.9 % when it was nearly
        unimodal, because a shallow Otsu optimum amplifies any perturbation.
      * Every image carries the same weight regardless of its pixel count,
        since each histogram is divided by its own pixel total. Otherwise a
        2048x1536 field would outvote a 1024x768 one four to one in setting a
        threshold that is then applied to both.
    """
    nbins = max(2, int(nbins))
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(nbins, dtype=np.float64)
    n_img = 0
    for g in grays:
        sm = prepare(g, p).ravel()
        if sm.size == 0:
            continue
        counts, _ = np.histogram(sm, bins=edges)
        hist += counts / float(sm.size)       # equal weight per image
        n_img += 1
    if n_img == 0:
        return float("nan")
    return _otsu_from_hist(hist, centers)


def analyze_array(gray: np.ndarray, p: Params, source: str = "",
                  forced_threshold: Optional[float] = None) -> Result:
    """
    Run the full pipeline on a float grayscale image in [0, 1].

    `forced_threshold` overrides the threshold that `p` would choose; it is how
    a batch-wide Otsu value is pushed into each image's analysis.
    """
    res = Result(source=source)
    h, w = gray.shape
    res.height_px, res.width_px = h, w
    res.field_w_um = w * p.pixel_size_um
    res.field_h_um = h * p.pixel_size_um
    res.field_area_um2 = res.field_w_um * res.field_h_um
    res.pixel_size_um = p.pixel_size_um
    res.sigma_px_used = p.resolved_sigma_px()
    res.opening_px_used = p.resolved_opening_px()

    # ---- how much brightness information is actually here ---------------
    rep = contrast_report(gray, p)
    res.dynamic_range = rep["dynamic_range"]
    res.gray_levels = rep["gray_levels"]
    res.norm_low_pct_used = rep["norm_low_pct"]

    # ---- normalize + smooth --------------------------------------------
    sm = prepare(gray, p)
    res.separability = otsu_separability(sm)

    # ---- threshold -----------------------------------------------------
    try:
        res.otsu_threshold = float(filters.threshold_otsu(sm))
    except Exception:
        res.otsu_threshold = float("nan")

    if forced_threshold is not None:
        thr = float(forced_threshold)
        res.threshold_source = "Otsu(일괄)"
    elif p.threshold_mode == "otsu":
        thr = res.otsu_threshold
        res.threshold_source = "Otsu(이미지별)"
    else:                                  # "fixed", and "otsu_batch" without
        thr = p.threshold                  # a pooled value supplied
        res.threshold_source = "고정"
    if not np.isfinite(thr):
        thr = p.threshold
        res.threshold_source = "고정(대체)"
    res.effective_threshold = float(thr)

    # ---- warnings the user must see, not bury in a column ---------------
    if np.isfinite(res.separability) and res.separability < p.min_separability:
        res.warnings.append(
            f"밝기 분리도 {res.separability:.2f} (<{p.min_separability:.2f}) — "
            "두 계급으로 갈라지지 않는 이미지입니다. 임계값이 사실상 임의로 "
            "정해지므로 이 행의 개공율은 신뢰하지 마십시오.")
    if res.gray_levels and res.gray_levels < 12:
        res.warnings.append(
            f"정규화 구간의 계조가 {res.gray_levels}단계뿐입니다 — "
            "대비를 높여 재촬영하시는 편이 낫습니다.")
    if (p.normalize_contrast and np.isfinite(res.norm_low_pct_used)
            and res.norm_low_pct_used < p.norm_low_pct - 1e-9):
        res.warnings.append(
            f"암부 면적이 좁아 정규화 하위 기준을 {p.norm_low_pct:.2f} % 대신 "
            f"{res.norm_low_pct_used:.2f} %로 낮춰 적용했습니다.")

    binary = sm < thr                     # dark = candidate opening
    res.dark_area_fraction = float(binary.mean())

    # ---- morphological opening ----------------------------------------
    open_px = p.resolved_opening_px()
    if open_px > 0:
        binary = morphology.opening(binary, morphology.disk(open_px))

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

        # The three area fractions partition the segmented pore area:
        #   total = valid (solidity >= cut) + invalid (solidity < cut)
        # `open_pore_fraction` is the valid one — the published metric.
        convex_mask = sol >= p.solidity_cut
        res.open_pore_fraction = float(area[convex_mask].sum() / res.field_area_um2)
        res.invalid_pore_fraction = float(area[~convex_mask].sum() / res.field_area_um2)
        res.total_pore_fraction = float(area.sum() / res.field_area_um2)
        res.concave_ratio = float((~convex_mask).sum() / len(kept))
        res.valid_count_ratio = 1.0 - res.concave_ratio
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


def unique_labels(paths) -> dict:
    """
    A short, unique label per path.

    A batch often spans folders that hold the same file names — cond_A/site01
    and cond_B/site01 — and showing both as "site01" makes the table and the
    CSV ambiguous. Parent folders are prepended only until the names separate.
    """
    import os
    paths = list(paths)
    labels = {p: os.path.basename(p) for p in paths}
    for _ in range(8):
        counts = {}
        for lab in labels.values():
            counts[lab] = counts.get(lab, 0) + 1
        clashing = [p for p in paths if counts[labels[p]] > 1]
        if not clashing:
            break
        grew = False
        for p in clashing:
            head = p
            for _ in range(labels[p].count(os.sep) + 1):
                head = os.path.dirname(head)
            parent = os.path.basename(head)
            if parent:
                labels[p] = os.path.join(parent, labels[p])
                grew = True
        if not grew:
            break
    return labels


def analyze_path(path: str, p: Params, label: str = None) -> Result:
    """Load an image file and analyze it."""
    gray = load_gray(path, crop_bottom_px=p.crop_bottom_px, crop_top_px=p.crop_top_px)
    import os
    res = analyze_array(gray, p, source=label or os.path.basename(path))
    res.source_path = path
    return res


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
        # the sweep is about the threshold VALUE, so pin the mode to fixed
        pt = p.copy_with(threshold=t, threshold_mode="fixed")
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
