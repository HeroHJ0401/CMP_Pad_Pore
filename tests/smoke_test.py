"""
CI smoke test: the pipeline must run end-to-end and behave sanely on a
synthetic image whose ground truth we control.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pore_analyzer.core import (Params, analyze_array, robustness_sweep,
                                detect_bands, pooled_otsu)
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
