"""
CI smoke test: the pipeline must run end-to-end and behave sanely on a
synthetic image whose ground truth we control.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pore_analyzer.core import (Params, analyze_array, robustness_sweep,
                                detect_bands, pooled_otsu, unique_labels)
from pore_analyzer.pixelsize import from_scale_bar
from tests.make_sample import make


def main():
    gray = make().astype(float) / 255.0
    p = Params()
    res = analyze_array(gray, p, source="synthetic")

    checks = [
        ("objects found", res.n_objects > 50),
        ("open-pore fraction in range", 0.0 < res.open_pore_fraction < 0.5),
        ("open <= dark area", res.open_pore_fraction <= res.dark_area_fraction + 1e-9),
        ("circularity sane", 0.0 < res.circularity_median <= 1.0),
        ("solidity sane", 0.0 < res.solidity_median <= 1.0),
        ("overlay shape", res.overlay.shape == gray.shape + (3,)),
        ("field size", abs(res.field_w_um - gray.shape[1] * 0.404) < 1e-6),
    ]

    # a more porous image must report a higher open-pore fraction
    dense = make(n_round=260, seed=7).astype(float) / 255.0
    res_dense = analyze_array(dense, p, source="dense")
    checks.append(("monotonic vs. porosity",
                   res_dense.open_pore_fraction > res.open_pore_fraction))

    # the sweep must produce the full grid without error
    rows = robustness_sweep(gray, p)
    checks.append(("sweep grid complete", len(rows) == 21))

    # Band detection: a letterboxed copy must measure the same as the original
    # once the bands are trimmed, and a clean image must not be trimmed at all.
    W = gray.shape[1]
    boxed = np.vstack([np.zeros((150, W)), gray, np.zeros((70, W))])
    top, bot = detect_bands(boxed)
    checks.append(("no band on a clean image", detect_bands(gray) == (0, 0)))
    checks.append(("letterbox found", (top, bot) == (150, 70)))
    trimmed = boxed[top: boxed.shape[0] - bot, :]
    res_trim = analyze_array(trimmed, p, source="trimmed")
    checks.append(("trimmed == original",
                   abs(res_trim.open_pore_fraction - res.open_pore_fraction) < 1e-9))
    res_boxed = analyze_array(boxed, p, source="boxed")
    checks.append(("untrimmed letterbox understates",
                   res_boxed.open_pore_fraction < res.open_pore_fraction))

    # ---- thresholding must survive a brightness/contrast difference -------
    # Two conditions that genuinely differ, where the more porous one was also
    # imaged brighter and flatter. The measured ratio must track the true one.
    lo = make(n_round=110, seed=3).astype(float) / 255.0
    hi = make(n_round=300, seed=4).astype(float) / 255.0
    hi_bright = np.clip(hi * 0.75 + 0.22, 0, 1)

    def ratio(a, b, **kw):
        q = Params(**kw)
        if q.threshold_mode == "otsu_batch":
            t = pooled_otsu([a, b], q)
            ra = analyze_array(a, q, forced_threshold=t)
            rb = analyze_array(b, q, forced_threshold=t)
        else:
            ra, rb = analyze_array(a, q), analyze_array(b, q)
        return rb.open_pore_fraction / ra.open_pore_fraction, ra, rb

    truth, _, _ = ratio(lo, hi, threshold_mode="fixed")
    fixed_shifted, _, _ = ratio(lo, hi_bright, threshold_mode="fixed")
    reco, ra, rb = ratio(lo, hi_bright, threshold_mode="otsu_batch",
                         normalize_contrast=True)

    checks.append(("fixed threshold is distorted by brightness",
                   abs(fixed_shifted - truth) / truth > 0.15))
    checks.append(("normalize + batch Otsu recovers the true ratio",
                   abs(reco - truth) / truth < 0.05))
    checks.append(("batch Otsu applies one threshold to every image",
                   abs(ra.effective_threshold - rb.effective_threshold) < 1e-12))
    checks.append(("effective threshold is recorded",
                   np.isfinite(ra.effective_threshold)))

    # ---- pixel size: what it does and does not touch ----------------------
    # The open-pore fraction divides an area by an area, so the pixel size
    # cancels. It reaches the number only through the min-diameter filter,
    # which is stated in µm. Getting this wrong would silently rescale a batch.
    free = Params(min_diam_um=0.0)
    invariant = [analyze_array(gray, free.copy_with(pixel_size_um=s)).open_pore_fraction
                 for s in (0.101, 0.404, 0.808)]
    checks.append(("open-pore fraction is scale-invariant",
                   max(invariant) - min(invariant) < 1e-12))

    filtered = [analyze_array(gray, Params(pixel_size_um=s)).open_pore_fraction
                for s in (0.101, 0.404)]
    checks.append(("min-diameter filter is what makes scale matter",
                   abs(filtered[0] - filtered[1]) > 1e-6))

    shapes = [analyze_array(gray, free.copy_with(pixel_size_um=s)).circularity_median
              for s in (0.101, 0.808)]
    checks.append(("circularity is dimensionless", abs(shapes[0] - shapes[1]) < 1e-12))

    # ---- physical units keep σ the same size at any magnification ---------
    # This is asserted as an exact property of the conversion, not as an
    # empirical improvement on some test image: what µm units guarantee is that
    # the smoothing and opening cover the same physical distance at every
    # magnification, so the filters mean the same thing in every image.
    phys = Params(sigma_um=0.5, opening_radius_um=0.8)
    same_physical_sigma = {
        round(phys.copy_with(pixel_size_um=s).resolved_sigma_px() * s, 9)
        for s in (0.101, 0.202, 0.404, 0.808)}
    checks.append(("σ in µm is the same physical size at every scale",
                   same_physical_sigma == {0.5}))
    checks.append(("σ in px is NOT the same physical size",
                   len({round(Params(pixel_size_um=s).resolved_sigma_px() * s, 9)
                        for s in (0.101, 0.404)}) == 2))
    checks.append(("opening in µm scales with the image",
                   phys.copy_with(pixel_size_um=0.404).resolved_opening_px() == 2
                   and phys.copy_with(pixel_size_um=0.202).resolved_opening_px() == 4))
    checks.append(("px values still work when µm is unset",
                   Params(sigma_px=1.2).resolved_sigma_px() == 1.2
                   and Params(opening_radius_px=2).resolved_opening_px() == 2))
    checks.append(("per-image scale is recorded in the result",
                   analyze_array(gray, Params(pixel_size_um=0.202)).pixel_size_um == 0.202))

    # ---- the three area fractions must partition the pore area ------------
    # Pore = 유효 + 무효. If this ever drifts the table is quietly lying, so it
    # is checked exactly rather than to a tolerance.
    checks.append(("total pore area = valid + invalid",
                   abs(res.total_pore_fraction
                       - (res.open_pore_fraction + res.invalid_pore_fraction)) < 1e-12))
    checks.append(("valid area is the published metric",
                   res.open_pore_fraction <= res.total_pore_fraction + 1e-12))
    checks.append(("count ratios sum to one",
                   abs(res.valid_count_ratio + res.concave_ratio - 1.0) < 1e-12))

    # ---- batch Otsu must not depend on the order the files arrive in ------
    # It used to: one shared random Generator subsampled each image, so the
    # draw for image k depended on how many came before it. Reordering the same
    # files moved the threshold and with it every reported number — worst on
    # sets whose pooled histogram has a shallow optimum, which is exactly when
    # the batch mode is reached for. Asserted as exact equality over every
    # permutation, because a tolerance would let the bug back in.
    import itertools
    imgs = []
    for gain, off in ((1.00, 0.00), (0.80, 0.06), (1.15, -0.05)):
        imgs.append(np.clip(gray * gain + off, 0, 1))
    pb = Params(normalize_contrast=True, threshold_mode="otsu_batch")
    seen = {round(pooled_otsu((imgs[i] for i in order), pb), 12)
            for order in itertools.permutations(range(len(imgs)))}
    checks.append(("batch Otsu identical for every file order", len(seen) == 1))
    # and equally weighted: blowing one image up to 4x the pixels must not let
    # it outvote the others in a threshold that is then applied to all of them.
    # Smoothing is off here on purpose — a Gaussian of a fixed pixel σ covers
    # half the physical distance on a 2x-upscaled image and would change its
    # histogram for real, which is a different effect from vote weighting.
    pw = Params(normalize_contrast=True, threshold_mode="otsu_batch", sigma_px=0.0)
    big = np.repeat(np.repeat(imgs[0], 2, axis=0), 2, axis=1)
    checks.append(("batch Otsu weighs each image equally, not by pixel count",
                   abs(pooled_otsu(iter(imgs), pw)
                       - pooled_otsu(iter([big] + imgs[1:]), pw)) < 1e-9))

    # ---- separability must flag an image Otsu cannot legitimately split ---
    # Otsu returns a threshold for any input, including a field with no pores:
    # it splits the noise and reports several percent out of nothing. eta is
    # what separates "two real classes" from "one blob cut in half", and it has
    # to stay high when contrast is merely low — otherwise it would just be
    # restating contrast and would fire on sound low-contrast captures.
    rng_s = np.random.default_rng(11)
    blank = np.clip(0.72 + rng_s.normal(0, 0.015, gray.shape), 0, 1)
    po = Params(threshold_mode="otsu", normalize_contrast=True)
    r_real = analyze_array(gray, po)
    r_blank = analyze_array(blank, po)
    checks.append(("real pores are separable", r_real.separability > 0.80))
    checks.append(("a pore-free field is flagged, not reported",
                   r_blank.separability < po.min_separability
                   and any("분리도" in w for w in r_blank.warnings)))
    checks.append(("a sound image carries no warning", not r_real.warnings))
    squashed = np.clip((gray - gray.mean()) * 0.06 + gray.mean(), 0, 1)
    squashed = np.round(squashed * 255) / 255
    r_low = analyze_array(squashed, po)
    checks.append(("separability survives a 16x contrast squeeze",
                   abs(r_low.separability - r_real.separability) < 0.05))
    checks.append(("low contrast is reported as a dynamic range, not an error",
                   r_low.dynamic_range < r_real.dynamic_range
                   and r_low.gray_levels < r_real.gray_levels))

    # ---- the normalization anchor must stay out of the dark population ----
    # With the anchor at 1 % and pores covering less than that, the percentile
    # lands in the bright matrix, matrix noise gets stretched across the range
    # and every pore is clipped to 0: a field whose true value was 0.41 % read
    # 5.16 %. The anchor is therefore held below the measured dark fraction.
    sparse = np.full(gray.shape, 0.72)
    yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
    for cy, cx in ((40, 50), (90, 140), (150, 60)):
        sparse[((yy - cy) ** 2 + (xx - cx) ** 2) <= 36] = 0.22
    sparse = np.clip(sparse + np.random.default_rng(5).normal(0, 0.015, gray.shape), 0, 1)
    true_dark = float((sparse < 0.45).mean())
    r_sp = analyze_array(sparse, Params(threshold_mode="otsu", normalize_contrast=True,
                                        min_diam_um=0.5))
    checks.append(("sparse dark population lowers the normalization anchor",
                   r_sp.norm_low_pct_used < 1.0
                   and any("하위 기준" in w for w in r_sp.warnings)))
    checks.append(("sparse field is not inflated by normalization",
                   r_sp.total_pore_fraction < 4 * true_dark + 0.01))
    # and the ordinary case must be untouched, so existing numbers still hold
    checks.append(("an ordinary field keeps the anchor as configured",
                   abs(r_real.norm_low_pct_used - 1.0) < 1e-9
                   and not any("하위 기준" in w for w in r_real.warnings)))

    # ---- three brightness populations need three classes ----------------
    # A worn surface holds deep pores, mid-grey shadow and bright matrix. Two-
    # class Otsu must draw one line through that and puts it between shadow and
    # matrix, so the shadow is scored as pore and fuses everything it touches.
    # Here the truth is known: pores at 0.20, shadow at 0.45, matrix at 0.80.
    rng3 = np.random.default_rng(17)
    three = np.full((300, 400), 0.80)
    three[:, :160] = 0.45                                 # the mid-grey band
    yy3, xx3 = np.ogrid[:300, :400]
    for cy, cx in ((70, 70), (150, 90), (220, 60), (90, 300), (200, 330)):
        three[((yy3 - cy) ** 2 + (xx3 - cx) ** 2) <= 30 ** 2] = 0.20
    three = np.clip(three + rng3.normal(0, 0.02, three.shape), 0, 1)
    p2 = Params(pixel_size_um=0.5, min_diam_um=1.0, threshold_mode="otsu")
    r2c = analyze_array(three, p2)
    r3c = analyze_array(three, p2.copy_with(threshold_mode="otsu3"))
    checks.append(("2-class Otsu lands between shadow and matrix",
                   0.45 < r2c.effective_threshold < 0.80))
    checks.append(("3-class Otsu lands between pore and shadow",
                   0.20 < r3c.effective_threshold < 0.45))
    checks.append(("so 2-class calls far more of the field dark",
                   r2c.dark_area_fraction > 2 * r3c.dark_area_fraction))
    checks.append(("the threshold's origin is reported",
                   r2c.threshold_source == "Otsu(이미지별)"
                   and r3c.threshold_source == "Otsu3(이미지별)"))
    checks.append(("both reference thresholds are always available",
                   np.isfinite(r2c.otsu_threshold) and np.isfinite(r2c.otsu3_threshold)
                   and r2c.otsu3_threshold < r2c.otsu_threshold))
    # the batch form has to keep the order-independence the 2-class one has
    imgs3 = [three, np.clip(three * 0.85 + 0.05, 0, 1),
             np.clip(three * 1.1 - 0.04, 0, 1)]
    pb3 = Params(normalize_contrast=True, threshold_mode="otsu3_batch")
    seen3 = {round(pooled_otsu((imgs3[i] for i in order), pb3), 12)
             for order in itertools.permutations(range(3))}
    checks.append(("3-class batch Otsu is identical for every file order",
                   len(seen3) == 1))
    checks.append(("3-class batch agrees with the single-image value on one image",
                   abs(pooled_otsu([three], Params(threshold_mode="otsu3_batch"))
                       - analyze_array(three, p2.copy_with(threshold_mode="otsu3")
                                       ).effective_threshold) < 0.01))

    # ---- fused pores must be separable, and only when asked -------------
    # Thin dark bridges between asperities fuse visually separate pores into one
    # blob. The fused blob is deeply concave, so it scores invalid, and if it
    # reaches the frame it is dropped whole — which looks exactly like "the
    # program never found these pores". A dumbbell is the minimal case.
    H2, W2 = 200, 320
    yy2, xx2 = np.ogrid[:H2, :W2]
    dumbbell = np.ones((H2, W2))
    for cx in (110, 210):
        dumbbell[((yy2 - 100) ** 2 + (xx2 - cx) ** 2) <= 38 ** 2] = 0.0
    dumbbell[96:105, 110:210] = 0.0                      # the neck
    ps = Params(pixel_size_um=0.5, sigma_px=0.0, opening_radius_px=0,
                threshold=0.5, min_diam_um=1.0, solidity_cut=0.90)
    r_join = analyze_array(dumbbell, ps)
    r_cut = analyze_array(dumbbell, ps.copy_with(objectify="distance",
                                                 split_depth_um=5.0))
    checks.append(("fused pores are one blob until splitting is asked for",
                   r_join.n_blobs_before_split == 1
                   and r_join.n_blobs_after_split == 1
                   and r_join.n_objects == 1))
    checks.append(("splitting turns the dumbbell into two pores",
                   r_cut.n_blobs_after_split == 2 and r_cut.n_objects == 2))
    checks.append(("the fused blob was invalid, the split ones are valid",
                   r_join.open_pore_fraction < r_cut.open_pore_fraction))
    checks.append(("splitting does not invent or lose dark area",
                   abs(r_join.dark_area_fraction - r_cut.dark_area_fraction) < 1e-12))
    # the depth is in µm, so it has to mean the same thing at any magnification
    coarse = analyze_array(dumbbell, ps.copy_with(objectify="distance",
                                                  split_depth_um=5.0,
                                                  pixel_size_um=1.0))
    checks.append(("split depth in µm behaves the same at another scale",
                   coarse.n_blobs_after_split == r_cut.n_blobs_after_split))

    # ---- objectification must not depend on the threshold being right ---
    # The point of "terrain": two pores separated by a ligament that the
    # threshold failed to keep above the cut. Connected components see one
    # object; flooding the grey image from its two basins sees two, because the
    # ridge between them is still a ridge whether or not it rose above the
    # threshold. Built so the ridge (0.44) sits BELOW a deliberately bad
    # threshold (0.55) — the case that broke the real image.
    H4, W4 = 220, 340
    yy4, xx4 = np.ogrid[:H4, :W4]
    bowls = np.full((H4, W4), 0.90)
    for cx in (120, 200):
        d4 = np.sqrt((yy4 - 110) ** 2 + (xx4 - cx) ** 2)
        bowls = np.minimum(bowls, 0.15 + 0.75 * np.clip(d4 / 60.0, 0, 1))
    # floors at 0.15, the saddle between them at 0.65, and a threshold of 0.72
    # deliberately set above the saddle so the mask fuses the two.
    pb_bad = Params(pixel_size_um=0.5, sigma_px=0.0, opening_radius_px=0,
                    threshold_mode="fixed", threshold=0.72, min_diam_um=1.0,
                    solidity_cut=0.0)
    r_cc = analyze_array(bowls, pb_bad)
    r_terr = analyze_array(bowls, pb_bad.copy_with(objectify="terrain",
                                                   terrain_depth=0.04))
    checks.append(("connected components fuse the two bowls",
                   r_cc.n_blobs_before_split == 1 and r_cc.n_objects == 1))
    checks.append(("terrain splits them despite the bad threshold",
                   r_terr.n_objects == 2))
    checks.append(("terrain does not change which pixels are dark",
                   abs(r_cc.dark_area_fraction - r_terr.dark_area_fraction) < 1e-12))
    checks.append(("terrain keeps the same total pore area",
                   abs(r_cc.total_pore_fraction - r_terr.total_pore_fraction) < 0.01))
    # a depth deeper than the basins themselves must fall back, not blow up
    r_deep = analyze_array(bowls, pb_bad.copy_with(objectify="terrain",
                                                   terrain_depth=5.0))
    checks.append(("an impossible terrain depth falls back to one object",
                   r_deep.n_objects == 1))

    # A solidity cut sitting on the median makes valid/invalid a coin flip.
    # Watershed cells carry straight cut edges, which lowers solidity
    # systematically, so this has to be flagged rather than reported straight.
    knife = analyze_array(gray, Params(threshold_mode="otsu", normalize_contrast=True,
                                       solidity_cut=0.0))
    on_cut = analyze_array(gray, Params(threshold_mode="otsu", normalize_contrast=True,
                                        solidity_cut=float(knife.solidity_median)))
    off_cut = analyze_array(gray, Params(threshold_mode="otsu", normalize_contrast=True,
                                         solidity_cut=float(knife.solidity_median) - 0.2))
    checks.append(("a solidity cut on the median is flagged",
                   any("거의 겹칩니다" in w for w in on_cut.warnings)))
    checks.append(("a cut well clear of the median is not flagged",
                   not any("거의 겹칩니다" in w for w in off_cut.warnings)))

    # ---- what was thrown away has to be visible, not silently absent ----
    edge = np.ones((160, 160))
    edge[0:40, 0:40] = 0.0                                # touches the frame
    edge[90:120, 90:120] = 0.0                            # does not
    pe = Params(pixel_size_um=0.5, sigma_px=0.0, opening_radius_px=0,
                threshold=0.5, min_diam_um=1.0)
    edge[140:144, 140:144] = 0.0                          # under the min diameter
    r_edge = analyze_array(edge, pe.copy_with(min_diam_um=2.5))
    checks.append(("frame-touching objects are reported as dropped area",
                   r_edge.dropped_border_fraction > 0.0))
    checks.append(("small objects are reported separately",
                   r_edge.dropped_small_fraction > 0.0))
    checks.append(("the two reasons add up to the total",
                   abs(r_edge.dropped_area_fraction
                       - (r_edge.dropped_border_fraction
                          + r_edge.dropped_small_fraction)) < 1e-12))
    # each reason must carry its OWN colour, or the user cannot check which
    # rule fired — a frame-touching blob can reach the middle of the field and
    # look nothing like "at the frame".
    ov = r_edge.overlay
    checks.append(("dropped area is drawn, not left blank",
                   ov is not None and bool((ov[5, 5] != ov[80, 5]).any())))
    checks.append(("the two drop reasons are drawn in different colours",
                   bool((ov[5, 5] != ov[141, 141]).any())))
    checks.append(("frame-touching is blue and small is not",
                   int(ov[5, 5][2]) > int(ov[5, 5][0])
                   and int(ov[141, 141][0]) > int(ov[5, 5][0])))
    r_keep = analyze_array(edge, pe.copy_with(exclude_border=False))
    checks.append(("nothing is dropped at the frame when nothing is excluded",
                   r_keep.dropped_border_fraction == 0.0))

    # ---- an SEM data bar carrying text must still be found --------------
    # The bar is not flat — it has white text on it — and a downscaled capture
    # blurs the glyph edges, which defeated both the flatness test and an
    # extreme-pixel count. A grey seam row at the very edge hid it completely.
    barred = np.vstack([gray, np.zeros((26, gray.shape[1]))])
    barred[-20:-14, 20:120] = 1.0                        # white text on the bar
    barred[-1, :] = 0.17                                 # blended edge row
    checks.append(("a data bar with text and a blended edge row is detected",
                   abs(detect_bands(barred)[1] - 26) <= 2))

    row = res.summary_row()
    checks.append(("CSV carries all three area fractions",
                   abs(row["valid_pore_fraction_pct"] + row["invalid_pore_fraction_pct"]
                       - row["pore_fraction_pct"]) < 0.02))
    checks.append(("non-pore is the complement of the dark area",
                   abs(row["non_pore_area_pct"] + row["dark_area_fraction_pct"]
                       - 100.0) < 1e-9))

    # an image with no objects at all must not divide by zero
    blank = np.full_like(gray, 0.9)
    res_blank = analyze_array(blank, p, source="blank")
    checks.append(("empty image is handled",
                   res_blank.n_objects == 0 and res_blank.open_pore_fraction == 0.0
                   and res_blank.total_pore_fraction == 0.0))

    # ---- labels must stay unique across folders ---------------------------
    import os as _os
    clash = [_os.path.join("x", "cond_A", "site01.tif"),
             _os.path.join("x", "cond_B", "site01.tif"),
             _os.path.join("x", "cond_A", "site02.tif")]
    labs = unique_labels(clash)
    checks.append(("duplicate basenames get distinct labels",
                   len(set(labs.values())) == 3))
    checks.append(("unique basenames stay short",
                   labs[clash[2]] == "site02.tif"))
    checks.append(("a single file needs no folder prefix",
                   unique_labels([_os.path.join("a", "b.tif")])[_os.path.join("a", "b.tif")]
                   == "b.tif"))

    # ---- scale bar arithmetic --------------------------------------------
    checks.append(("scale bar µm/px", abs(from_scale_bar(250, 100, "µm") - 0.4) < 1e-12))
    checks.append(("scale bar unit conversion",
                   abs(from_scale_bar(250, 100000, "nm") - 0.4) < 1e-9))
    bad = 0
    for args in ((0, 100, "µm"), (250, 0, "µm"), (250, 100, "furlong")):
        try:
            from_scale_bar(*args)
        except ValueError:
            bad += 1
    checks.append(("scale bar rejects bad input", bad == 3))

    print(f"\ntrue ratio {truth:.2f} | fixed under brightness shift "
          f"{fixed_shifted:.2f} | normalize+batch Otsu {reco:.2f}")

    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= bool(passed)

    print(f"\nopen-pore fraction: {100*res.open_pore_fraction:.2f} % "
          f"(dense image: {100*res_dense.open_pore_fraction:.2f} %)")
    print(f"objects: {res.n_objects}, circularity median {res.circularity_median:.3f}, "
          f"solidity median {res.solidity_median:.3f}")

    if not ok:
        print("\nSMOKE TEST FAILED", file=sys.stderr)
        return 1
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
