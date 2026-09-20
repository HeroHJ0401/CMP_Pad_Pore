"""Generate a synthetic pad-like SEM image for testing (no real data involved)."""

import sys
import numpy as np
from PIL import Image


def make(h=620, w=884, n_round=120, n_torn=30, seed=1):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 0.72)
    yy, xx = np.mgrid[0:h, 0:w]

    for _ in range(n_round):                       # convex openings
        cy, cx = rng.integers(20, h - 20), rng.integers(20, w - 20)
        r = rng.integers(5, 14)
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 0.15

    for _ in range(n_torn):                        # concave / occluded shapes
        cy, cx = rng.integers(20, h - 20), rng.integers(20, w - 20)
        r = rng.integers(8, 18)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        m &= ~(((yy - cy + r // 2) ** 2 + (xx - cx) ** 2) < (r * 0.8) ** 2)
        img[m] = 0.18

    img = np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1)
    return (img * 255).astype(np.uint8)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample.png"
    Image.fromarray(make()).save(out)
    print(f"wrote {out}")
