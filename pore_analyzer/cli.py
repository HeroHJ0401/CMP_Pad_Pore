"""Headless command-line interface — same algorithm as the GUI."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

from PIL import Image

from .core import (Params, analyze_array, load_gray, robustness_sweep,
                   detect_bands, pooled_otsu, unique_labels)
from .pixelsize import read_pixel_size, format_pixel_size


def build_parser():
    ap = argparse.ArgumentParser(
        prog="pore-analyzer",
        description="CMP pad SEM 이미지의 유효 Pore 개공율(open-pore fraction)을 산출합니다.")
    ap.add_argument("images", nargs="+", help="이미지 파일 또는 glob 패턴")
    ap.add_argument("--pixel-size", type=float, default=0.404, help="µm/px (기본 0.404)")
    ap.add_argument("--sigma", type=float, default=1.2, help="Gaussian σ, px (기본 1.2)")
    ap.add_argument("--threshold", type=float, default=0.45, help="이진화 임계 0–1 (기본 0.45)")
    ap.add_argument("--min-diam", type=float, default=2.0, help="최소 등가직경 µm (기본 2.0)")
    ap.add_argument("--solidity", type=float, default=0.90, help="Solidity 기준 (기본 0.90)")
    ap.add_argument("--opening-radius", type=int, default=2, help="Opening 반경 px (기본 2)")
    ap.add_argument("--crop-bottom", type=int, default=0, help="하단 크롭 px (기본 0, 0이면 자동 감지)")
    ap.add_argument("--crop-top", type=int, default=0, help="상단 크롭 px (기본 0, 0이면 자동 감지)")
    ap.add_argument("--no-autocrop", action="store_true", help="위·아래 단색 띠 자동 크롭 비활성화")
    ap.add_argument("--keep-border", action="store_true", help="프레임 접촉 객체를 제외하지 않음")
    ap.add_argument("--mode", choices=["fixed", "otsu", "otsu-batch"], default="fixed",
                    help="임계 결정 방식 (기본 fixed). otsu-batch는 모든 이미지의 "
                         "밝기 분포를 합쳐 공통 임계값 하나를 적용")
    ap.add_argument("--normalize", action="store_true",
                    help="이미지마다 대비를 정규화한 뒤 임계 적용 (밝기 차이 상쇄)")
    ap.add_argument("--auto-pixel-size", action="store_true",
                    help="파일 메타데이터에서 픽셀 크기를 읽어 --pixel-size 대신 사용")
    ap.add_argument("--sigma-um", type=float, default=None,
                    help="평활 σ를 µm로 지정 (--sigma 대신). 배율이 다른 이미지를 "
                         "함께 분석할 때 물리적 기준을 맞춤")
    ap.add_argument("--opening-um", type=float, default=None,
                    help="Opening 반경을 µm로 지정 (--opening-radius 대신)")
    ap.add_argument("--csv", metavar="PATH", help="결과를 CSV로 저장")
    ap.add_argument("--overlay-dir", metavar="DIR", help="오버레이 PNG를 저장할 디렉터리")
    ap.add_argument("--sweep", action="store_true", help="임계×solidity 강건성 스윕도 출력")
    return ap


def expand(patterns):
    out = []
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            out.extend(sorted(hits))
        elif os.path.exists(pat):
            out.append(pat)
        else:
            print(f"[경고] 찾을 수 없음: {pat}", file=sys.stderr)
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    files = expand(args.images)
    if not files:
        print("분석할 이미지가 없습니다.", file=sys.stderr)
        return 1

    base = Params(
        pixel_size_um=args.pixel_size,
        sigma_px=args.sigma,
        threshold=args.threshold,
        min_diam_um=args.min_diam,
        solidity_cut=args.solidity,
        opening_radius_px=args.opening_radius,
        crop_bottom_px=args.crop_bottom,
        crop_top_px=args.crop_top,
        exclude_border=not args.keep_border,
        normalize_contrast=args.normalize,
        threshold_mode=args.mode.replace("-", "_"),
        sigma_um=args.sigma_um,
        opening_radius_um=args.opening_um,
    )

    # settle cropping (and, if asked, pixel size) per file before anything else
    plans = []
    for path in files:
        p = base
        if not args.no_autocrop and base.crop_bottom_px == 0 and base.crop_top_px == 0:
            top, bot = detect_bands(load_gray(path))
            if top or bot:
                p = p.copy_with(crop_top_px=top, crop_bottom_px=bot)
                print(f"  ({os.path.basename(path)}: 단색 띠 자동 크롭 "
                      f"위 {top} px, 아래 {bot} px)")
        if args.auto_pixel_size:
            got = read_pixel_size(path)
            if got:
                p = p.copy_with(pixel_size_um=got[0])
                print(f"  ({os.path.basename(path)}: 픽셀 크기 "
                      f"{format_pixel_size(got[0])} [{got[1]}])")
            else:
                print(f"  ({os.path.basename(path)}: 메타데이터 없음 — "
                      f"--pixel-size {p.pixel_size_um} 사용)")
        plans.append((path, p))

    # Mixed magnifications are allowed, but say so: the open-pore fraction is
    # scale-invariant while σ and opening are not, unless given in µm.
    scales = {round(p.pixel_size_um, 6) for _path, p in plans}
    if len(scales) > 1:
        print(f"\n[주의] 배율 혼재 — 픽셀 크기가 {min(scales):.4g}–{max(scales):.4g} "
              f"µm/px로 {len(scales)}종입니다.")
        if base.sigma_um is None or base.opening_radius_um is None:
            print("       σ·opening이 픽셀 단위라 이미지마다 물리적 크기가 다릅니다. "
                  "--sigma-um / --opening-um 으로 지정하시면 기준이 같아집니다.")

    forced = None
    if base.threshold_mode == "otsu_batch":
        def grays():
            for path, p in plans:
                yield load_gray(path, p.crop_bottom_px, p.crop_top_px)
        forced = pooled_otsu(grays(), base)
        if forced == forced:
            print(f"\n공통 임계값 {forced:.4f} — {len(plans)}장 전부에 동일 적용")
        else:
            forced = None
            print("\n공통 임계값을 구하지 못해 고정 임계값을 사용합니다.")

    # Batches often span folders holding the same file names; label them so
    # the console output and the CSV stay unambiguous.
    labels = unique_labels([path for path, _ in plans])

    rows = []
    for path, p in plans:
        gray = load_gray(path, p.crop_bottom_px, p.crop_top_px)
        res = analyze_array(gray, p, source=labels[path], forced_threshold=forced)
        res.source_path = path
        rows.append(res.summary_row())

        print(f"\n=== {res.source} ===")
        print(f"  시야           {res.field_w_um:.1f} x {res.field_h_um:.1f} µm")
        print(f"  Pore 수        {res.n_objects}  (밀도 {res.object_density_per_mm2:.0f} /mm²)")
        print(f"  픽셀 크기      {p.pixel_size_um:.5g} µm/px")
        print(f"  적용 임계      {res.effective_threshold:.4f}  "
              f"(모드 {p.threshold_mode}"
              f"{', 대비 정규화' if p.normalize_contrast else ''}"
              f", 이 이미지의 Otsu {res.otsu_threshold:.3f})")
        print(f"  유효 Pore 개공율 {100*res.open_pore_fraction:.2f} %   <-- 주 지표")
        print(f"  Pore 개공율    {100*res.total_pore_fraction:.2f} % "
              f"(= 유효 {100*res.open_pore_fraction:.2f} + "
              f"무효 {100*res.invalid_pore_fraction:.2f})")
        print(f"  Pore 면적(필터전) {100*res.dark_area_fraction:.2f} %   "
              f"(Non-Pore 면적 {100*(1-res.dark_area_fraction):.2f} %)")
        print(f"  Pore 원형도    {res.circularity_median:.3f} "
              f"(IQR {res.circularity_q1:.3f}–{res.circularity_q3:.3f})")
        print(f"  유효 Pore 원형도 {res.solidity_median:.3f} "
              f"(IQR {res.solidity_q1:.3f}–{res.solidity_q3:.3f})")
        print(f"  유효/무효 Pore 비율 {100*res.valid_count_ratio:.1f} % / "
              f"{100*res.concave_ratio:.1f} %  (개수 기준)")
        print(f"  등가직경 중앙값 {res.eqdiam_median_um:.2f} µm")
        print(f"  제외: 소형 {res.n_rejected_small}, 프레임접촉 {res.n_rejected_border} "
              f"(면적 {100*res.border_area_fraction:.1f} %)")

        if args.overlay_dir:
            # Name from the disambiguated label, or two folders holding the
            # same file name would silently overwrite each other's overlay.
            stem = os.path.splitext(labels[path])[0].replace(os.sep, "_")
            out = os.path.join(args.overlay_dir, f"{stem}_overlay.png")
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            Image.fromarray(res.overlay).save(out)
            print(f"  오버레이 저장  {out}")

        if args.sweep:
            print("  --- 강건성 스윕 (임계 x solidity -> 유효 Pore 개공율 %) ---")
            for r in robustness_sweep(load_gray(path, p.crop_bottom_px, p.crop_top_px), p):
                print(f"    thr={r['threshold']:.2f}  S={r['solidity_cut']:.2f}  "
                      f"{r['open_pore_fraction_pct']:6.2f} %   n={r['n_objects']}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV 저장: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
