"""
CI smoke test: the pipeline must run end-to-end and behave sanely on a
synthetic image whose ground truth we control.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pore_analyzer.core import Params, analyze_array, robustness_sweep, detect_bands
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
