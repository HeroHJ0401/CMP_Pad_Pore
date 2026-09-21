"""
Tkinter GUI for CMP pad SEM open-pore fraction analysis.

Drag images onto the window (if tkinterdnd2 is available), drop them on the
.exe icon, or use the 이미지 추가 button. Results appear in a table and the
segmentation overlay is drawn on the right. Hover any column header or any
analysis-condition label for a description of what it means.
"""

from __future__ import annotations

import os
import sys
import csv
import threading
import traceback
import queue

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from .core import (Params, load_gray, analyze_array, robustness_sweep,
                   detect_bands, pooled_otsu, unique_labels)
from .pixelsize import read_pixel_size, format_pixel_size, ScaleStore
from .scalebar import measure_scale_bar

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:
    HAS_DND = False

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

CONVEX = "#2ecc71"      # solidity >= cut  -> counted in the open-pore fraction
CONCAVE = "#e74c3c"     # solidity <  cut  -> excluded
PANEL_W = 460           # preview panel width; also the label wrap width
APP_NAME = "CMP Pad 개공율 분석기_정현진"

# key, heading, width, tooltip
COLS = [
    ("file", "파일", 160,
     "분석한 이미지 파일명입니다."),
    ("pixel_size_um", "µm/px", 70,
     "이 이미지에 적용된 픽셀 크기입니다. 이미지마다 다를 수 있습니다.\n\n"
     "개공률은 분자와 분모가 모두 픽셀 크기의 제곱으로 스케일되므로 이 값에 "
     "거의 영향을 받지 않습니다. 영향을 받는 것은 등가직경·밀도·시야, 그리고 "
     "최소 등가직경(µm) 필터를 통과하는 객체의 범위입니다.\n\n"
     "원형도와 solidity는 무차원이라 전혀 영향이 없습니다."),
    ("n_objects", "객체수", 62,
     "모든 필터를 통과해 형상 지표 계산에 사용된 어두운 영역의 개수입니다.\n\n"
     "제외되는 것: 등가직경이 최소값 미만인 객체, 이미지 경계에 닿은 객체."),
    ("open_pore_fraction_pct", "개공률 %", 80,
     "주 지표입니다.\n\n"
     "Solidity가 기준값 이상인 볼록 개구부들의 면적 합 ÷ 시야 전체 면적 × 100.\n\n"
     "어두운 영역에는 노출된 기공과 표면 그림자가 섞여 있는데, 그림자 덩어리는 "
     "대체로 오목하므로 solidity 기준으로 걸러냅니다. 그래서 단순 암부면적률과는 "
     "값도 방향도 다를 수 있습니다.\n\n"
     "주의: 임계값과 solidity 기준에 따라 절대값이 크게 움직입니다. 동일 설정에서 "
     "두 조건을 비교한 대소 관계와 비율만 사용하십시오."),
    ("dark_area_fraction_pct", "암부면적 %", 88,
     "평활 후 이진화 임계값 미만인 픽셀의 비율입니다. 참고용입니다.\n\n"
     "노출된 기공과 표면 그림자가 모두 포함되므로, 표면이 거칠수록 커지는 "
     "거칠기 대리 지표에 가깝습니다. 이 값을 개공률로 쓰면 결론이 뒤집힐 수 "
     "있습니다."),
    ("circularity_median", "원형도 중앙값", 98,
     "원형도 = 4πA / P²  (A: 면적, P: 둘레)\n\n"
     "완전한 원이면 1, 길쭉하거나 가장자리가 복잡할수록 0에 가까워집니다. "
     "전체 객체의 원형도를 구한 뒤 그 중앙값을 표시합니다.\n\n"
     "값이 높을수록 개구부가 덜 폐색되어 원래 형태를 유지하고 있음을 시사합니다."),
    ("solidity_median", "Solidity 중앙값", 108,
     "Solidity = A / A_convex hull  (볼록 껍질 면적 대비 실제 면적)\n\n"
     "1에 가까우면 오목한 결손이 없는 볼록한 형태이고, 낮을수록 개구부가 "
     "무너진 asperity에 가려져 오목하게 잘려 나갔음을 뜻합니다.\n\n"
     "개공률 분자에 포함할지 여부를 이 값으로 판정합니다."),
    ("eqdiam_median_um", "등가직경 µm", 92,
     "각 객체와 같은 면적을 갖는 원의 지름으로 환산한 값의 중앙값입니다.\n\n"
     "등가직경 = 2 × √(A / π), 픽셀 크기를 곱해 µm로 변환합니다."),
    ("object_density_per_mm2", "밀도 /mm²", 84,
     "객체수 ÷ 시야 면적(mm²).\n\n"
     "시야 면적은 (가로 픽셀 × 픽셀 크기) × (세로 픽셀 × 픽셀 크기)이며, "
     "크롭된 영역은 제외됩니다."),
    ("concave_ratio_pct", "오목비율 %", 84,
     "Solidity가 기준값 미만인 객체의 개수 비율입니다.\n\n"
     "면적이 아니라 개수 기준이며, 폐색된 개구부가 얼마나 흔한지를 나타냅니다."),
    ("n_rejected_border", "프레임접촉", 80,
     "이미지 경계에 닿아 제외된 객체의 개수입니다.\n\n"
     "잘린 객체는 둘레와 볼록 껍질이 실제와 달라 원형도·solidity를 신뢰할 수 "
     "없으므로 제외합니다. ('프레임 접촉 객체 제외'를 끄면 포함됩니다.)"),
    ("border_area_fraction_pct", "접촉면적 %", 88,
     "프레임 접촉으로 제외된 객체들이 차지하던 면적의 비율입니다.\n\n"
     "큰 객체일수록 경계에 닿을 확률이 높으므로, 이 값이 크면 개공률은 그만큼 "
     "과소평가된 상태입니다. 논문에 수치를 쓰실 때 함께 밝히시는 것이 정직합니다."),
    ("effective_threshold", "적용 임계", 78,
     "이 이미지에 실제로 적용된 이진화 임계값입니다.\n\n"
     "'고정'에서는 입력하신 값 그대로이고, 'Otsu(이미지별)'에서는 이미지마다 "
     "다르며, 'Otsu(일괄)'에서는 모든 이미지가 같은 값을 갖습니다.\n\n"
     "두 조건을 비교하실 때는 이 열의 값이 서로 같은지 반드시 확인하십시오. "
     "값이 다르면 개공률 차이에 임계값 차이가 섞여 들어갑니다."),
    ("otsu_threshold_ref", "Otsu(참고)", 84,
     "이 이미지 한 장만으로 산출한 Otsu 임계값입니다. 모드가 "
     "'Otsu(이미지별)'일 때만 실제로 쓰이고, 그 외에는 참고용입니다.\n\n"
     "설정하신 고정 임계값이 이 값과 크게 다르면, 밝기나 대비가 다른 이미지를 "
     "같은 고정 임계로 비교하고 있다는 신호입니다. 그럴 때는 '대비 정규화'를 "
     "켜시거나 'Otsu(일괄)'로 바꾸십시오."),
]

PARAM_HELP = {
    "pixel_size_um":
        "이미지 한 픽셀이 실제로 몇 µm인지입니다.\n\n"
        "직접 계산하실 필요 없습니다. 이미지를 추가하시면 파일 메타데이터에서 "
        "자동으로 읽어 채웁니다(Zeiss·FEI·ImageJ·TIFF 해상도 태그). 메타데이터가 "
        "없는 화면 캡처나 JPEG라면 옆의 '스케일바로 측정' 버튼을 누르고 이미지에 "
        "찍힌 스케일바를 드래그하신 뒤 거기 적힌 길이를 입력하시면 됩니다.\n\n"
        "면적·등가직경·밀도·개공률이 모두 이 값에 비례합니다. 형상 지표"
        "(원형도·solidity)는 영향을 받지 않습니다.",
    "sigma_px":
        "이진화 전에 적용하는 가우시안 평활의 표준편차(픽셀)입니다.\n\n"
        "각 픽셀은 자기 밝기가 아니라 주변 약 ±3σ 이웃의 가중평균과 임계값을 "
        "비교하게 됩니다. 픽셀 노이즈가 작은 점 객체로 잡히는 것을 막습니다.\n\n"
        "0을 넣으면 평활을 건너뜁니다.",
    "threshold":
        "정규화 밝기(0~1) 기준으로 이 값보다 어두운 픽셀을 개구부 후보로 봅니다.\n\n"
        "임계 모드가 '고정'일 때만 쓰입니다. Otsu 모드에서는 프로그램이 임계값을 "
        "직접 고르므로 이 칸은 무시됩니다.\n\n"
        "값 자체의 타당성은 결과 표의 'Otsu(참고)' 값과 대조해 보십시오.",
    "min_diam_um":
        "등가직경이 이 값보다 작은 덩어리는 버립니다. 유효 최소 스케일에 해당합니다.\n\n"
        "기본값 2.0 µm는 픽셀 크기 0.404 µm 기준으로 약 5픽셀, 면적으로는 약 19픽셀입니다.",
    "solidity_cut":
        "개공률 분자에 포함할 볼록도의 하한입니다.\n\n"
        "이 값 이상인 객체의 면적만 합산합니다. 낮추면 그림자성 오목 영역이 섞여 "
        "들어오고, 높이면 실제 개구부까지 탈락합니다.",
    "opening_radius_px":
        "이진화 후 적용하는 형태학적 opening의 disk 반경입니다.\n\n"
        "폭이 대략 2r 픽셀 미만인 가는 연결부를 끊어, 서로 다른 개구부가 한 덩어리로 "
        "붙는 것을 막습니다. 0을 넣으면 건너뜁니다.",
    "crop_bottom_px":
        "이미지 아래쪽에서 잘라낼 행 수입니다. SEM 정보바 제거용입니다.\n\n"
        "0으로 두고 '자동 크롭'을 켜두시면 프로그램이 감지해 잘라냅니다. 자동 감지가 "
        "놓치거나 과하게 자를 때만 직접 지정하십시오.",
    "crop_top_px":
        "이미지 위쪽에서 잘라낼 행 수입니다. 화면 캡처의 검은 여백(레터박스) 제거용입니다.\n\n"
        "잘라내지 않으면 그 띠가 시야 면적에 포함되어 개공률이 실제보다 낮게 나옵니다.",
}

THRESHOLD_MODES = [
    ("고정", "fixed"),
    ("Otsu (이미지별)", "otsu"),
    ("Otsu (일괄)", "otsu_batch"),
]

MODE_HELP = (
    "어두운 곳과 밝은 곳을 가르는 기준을 어떻게 정할지입니다.\n\n"
    "• 고정 — 입력하신 임계값을 그대로 씁니다. 촬영 조건이 완전히 동일한 "
    "이미지들에만 안전합니다. 한쪽이 더 밝게 찍히면 그 차이가 개공률 차이로 "
    "둔갑합니다.\n\n"
    "• Otsu (이미지별) — 이미지마다 밝기 분포를 보고 임계값을 따로 정합니다. "
    "밝기 차이는 사라지지만, 조건 간 실제 차이까지 일부 흡수할 수 있습니다.\n\n"
    "• Otsu (일괄) — 불러온 모든 이미지의 밝기 분포를 합쳐 임계값 하나를 정하고 "
    "전부에 똑같이 적용합니다. 이미지 집합의 실제 밝기 범위에 맞추면서도 "
    "비교 대상 간에는 동일한 기준이 유지됩니다.\n\n"
    "두 조건을 비교하신다면 '대비 정규화 + Otsu(일괄)'을 권합니다. "
    "'권장 설정' 버튼으로 한 번에 맞출 수 있습니다."
)

CHECK_HELP = {
    "phys":
        "평활 σ와 opening 반경을 픽셀이 아니라 µm로 지정합니다.\n\n"
        "배율이 다른 이미지를 함께 분석하실 때 필요합니다. σ = 1.2 px는 "
        "0.404 µm/px에서 0.485 µm이지만 0.101 µm/px에서는 0.121 µm라, "
        "같은 숫자가 전혀 다른 물리적 크기가 됩니다.\n\n"
        "켜시면 입력하신 µm 값을 이미지마다 자기 픽셀 크기로 나누어 픽셀 수를 "
        "구하므로, 배율이 달라도 평활과 opening이 같은 물리적 거리를 덮습니다. "
        "즉 필터가 모든 이미지에서 같은 의미를 갖습니다.\n\n"
        "이것이 보장하는 것은 기준의 일관성이지, 개공률이 반드시 더 가까워진다는 "
        "뜻은 아닙니다. 실제 차이는 표면과 배율에 따라 달라집니다.",
    "normalize":
        "이미지마다 밝기·대비를 자기 자신의 범위에 맞춰 늘려 편 뒤에 임계를 "
        "적용합니다.\n\n"
        "하위 1 %와 상위 99 % 밝기를 각각 0과 1로 보내는 방식이라, SEM의 "
        "brightness/contrast 설정이 달라 생긴 차이를 상쇄합니다. 극단값을 기준으로 "
        "삼기 때문에 기공이 많고 적음에는 거의 흔들리지 않습니다.\n\n"
        "합성 이미지 실험에서, 한쪽을 밝고 대비 낮게 찍은 경우 고정 임계만으로는 "
        "조건 간 비율이 2.25배에서 1.73배로 무너졌지만, 이 옵션을 켜면 2.22배로 "
        "돌아왔습니다.",
    "border":
        "이미지 경계에 닿은 객체를 계산에서 제외합니다.\n\n"
        "잘린 객체는 둘레와 볼록 껍질이 실제와 달라 형상 지표를 신뢰할 수 없습니다. "
        "다만 큰 객체일수록 경계에 닿기 쉬우므로, 제외하면 개공률은 과소평가됩니다. "
        "제외된 면적 비율은 결과 표의 '접촉면적 %'에서 확인하십시오.",
    "autocrop":
        "이미지 위아래의 단색 띠를 자동으로 찾아 잘라냅니다.\n\n"
        "SEM 정보바와 화면 캡처의 검은 여백이 대상입니다. 거의 단색이면서 "
        "완전한 흑/백에 가까운 띠만 인정하는 보수적인 판정이라, 실제 패드 표면을 "
        "잘라낼 위험은 낮습니다. 크롭 값을 직접 입력하시면 그쪽이 우선합니다.",
}


# --------------------------------------------------------------------------
# Tooltip
# --------------------------------------------------------------------------

class Tooltip:
    """One reusable popup. `show` is idempotent for the same key."""

    def __init__(self, root, delay=400, wraplength=420):
        self.root = root
        self.delay = delay
        self.wraplength = wraplength
        self._win = None
        self._after = None
        self._key = None

    def request(self, key, text, x, y):
        if key == self._key:
            return                       # already showing / already scheduled
        self.hide()
        self._key = key
        self._after = self.root.after(self.delay, lambda: self._show(text, x, y))

    def hide(self):
        if self._after is not None:
            try:
                self.root.after_cancel(self._after)
            except Exception:
                pass
            self._after = None
        if self._win is not None:
            try:
                self._win.destroy()
            except Exception:
                pass
            self._win = None
        self._key = None

    def _show(self, text, x, y):
        self._after = None
        win = tk.Toplevel(self.root)
        win.wm_overrideredirect(True)
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass
        frm = tk.Frame(win, background="#2b2b2b", borderwidth=0)
        frm.pack()
        tk.Label(frm, text=text, justify="left", wraplength=self.wraplength,
                 background="#2b2b2b", foreground="#f0f0f0",
                 font=("Segoe UI", 9), padx=10, pady=8).pack()
        win.update_idletasks()

        # keep the popup on screen
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = win.winfo_width(), win.winfo_height()
        x = min(max(x, 0), max(sw - w - 4, 0))
        if y + h + 8 > sh:
            y -= h + 24
        win.wm_geometry(f"+{int(x)}+{int(y)}")
        self._win = win


PREVIEW_MAX = 1000      # longest side kept in memory for the overlay preview


def _shrink_overlay(overlay, max_side=PREVIEW_MAX):
    """Downscale an overlay to preview size; the original is not kept."""
    if overlay is None:
        return None
    h, w = overlay.shape[:2]
    if max(h, w) <= max_side:
        return overlay
    scale = max_side / float(max(h, w))
    im = Image.fromarray(overlay).resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.asarray(im)


def attach_tip(widget, tooltip: Tooltip, text, key=None):
    """Static tooltip for one widget."""
    key = key or (str(widget), text[:24])

    def enter(e):
        tooltip.request(key, text, e.x_root + 14, e.y_root + 18)

    def leave(_):
        tooltip.hide()

    widget.bind("<Enter>", enter, add="+")
    widget.bind("<Leave>", leave, add="+")
    widget.bind("<ButtonPress>", leave, add="+")


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

class App:
    def __init__(self, root, initial_files=None):
        self.root = root
        root.title(APP_NAME)
        root.geometry("1320x840")
        root.minsize(1100, 700)

        self.files: list[str] = []
        self.results: dict[str, object] = {}
        # path -> (um_per_px or None, where it came from)
        self.px_info: dict[str, tuple] = {}
        self.scale_store = ScaleStore()
        self._failures: list = []
        self._preview_img = None
        self._q: queue.Queue = queue.Queue()
        self._busy = False
        self.tip = Tooltip(root)

        self._build_style()
        self._build_widgets()

        # Drag-and-drop is a convenience, never a requirement: tkinterdnd2 can
        # import while its tkdnd binaries fail to load in a frozen build, and a
        # plain Tk root has no drop_target_register at all.
        self.dnd_ok = False
        if HAS_DND:
            try:
                root.drop_target_register(DND_FILES)
                root.dnd_bind("<<Drop>>", self._on_drop)
                self.dnd_ok = True
            except Exception:
                self.dnd_ok = False
        self._set_drop_hint()

        if initial_files:
            self._add_files(initial_files)

        self.root.after(80, self._drain_queue)

    # ------------------------------------------------------------------ UI
    def _build_style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("Treeview", rowheight=24)
        s.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        s.configure("Hint.TLabel", foreground="#666")
        s.configure("Big.TLabel", font=("Segoe UI", 15, "bold"))
        s.configure("Legend.TLabel", font=("Segoe UI", 9))
        s.configure("Warn.TLabel", foreground="#b8860b", font=("Segoe UI", 9))

    def _build_widgets(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")

        # ---- file list -------------------------------------------------
        fbox = ttk.LabelFrame(top, text="이미지", padding=6)
        fbox.pack(side="left", fill="both", expand=True)

        # A table rather than a plain list: every image carries its own scale,
        # because a batch may mix magnifications.
        fholder = tk.Frame(fbox, width=420, height=132)
        fholder.pack_propagate(False)
        fholder.pack(side="left", fill="both", expand=True)
        self.flist = ttk.Treeview(fholder, columns=("name", "px", "src"),
                                  show="headings", height=6, selectmode="extended")
        self.flist.heading("name", text="파일")
        self.flist.heading("px", text="µm/px")
        self.flist.heading("src", text="출처")
        self.flist.column("name", width=190, anchor="w", stretch=True)
        self.flist.column("px", width=78, anchor="center", stretch=False)
        self.flist.column("src", width=120, anchor="w", stretch=False)
        self.flist.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(fholder, orient="vertical", command=self.flist.yview)
        sb.pack(side="left", fill="y")
        self.flist.config(yscrollcommand=sb.set)
        self.flist.bind("<<TreeviewSelect>>", lambda e: self._show_preview())
        self.flist.bind("<Double-1>", self._edit_pixel_size)
        self.flist.tag_configure("unknown", foreground="#b8860b")

        fbtn = ttk.Frame(fbox)
        fbtn.pack(side="left", fill="y", padx=(8, 0))
        ttk.Button(fbtn, text="이미지 추가", command=self._browse).pack(fill="x", pady=2)
        ttk.Button(fbtn, text="선택 제거", command=self._remove_sel).pack(fill="x", pady=2)
        ttk.Button(fbtn, text="전체 비우기", command=self._clear).pack(fill="x", pady=2)
        b_px = ttk.Button(fbtn, text="픽셀 크기 지정…", command=self._edit_pixel_size)
        b_px.pack(fill="x", pady=(8, 2))
        attach_tip(b_px, self.tip,
                   "선택한 이미지의 픽셀 크기를 직접 입력하거나 스케일바로 "
                   "측정합니다. 표의 행을 더블클릭하셔도 같습니다.\n\n"
                   "배율이 서로 다른 이미지를 함께 분석하실 수 있습니다. "
                   "이미지마다 자기 값을 갖습니다.",
                   key="btn:pxedit")
        self.lbl_hint = ttk.Label(fbtn, text="", style="Hint.TLabel", wraplength=130)
        self.lbl_hint.pack(fill="x", pady=(6, 0))

        # ---- parameters ------------------------------------------------
        pbox = ttk.LabelFrame(top, text="분석 조건  (항목에 마우스를 올리면 설명이 나옵니다)",
                              padding=6)
        pbox.pack(side="left", fill="y", padx=(8, 0))

        self.v = {
            "pixel_size_um": tk.StringVar(value="0.404"),
            "sigma_px": tk.StringVar(value="1.2"),
            "threshold": tk.StringVar(value="0.45"),
            "min_diam_um": tk.StringVar(value="2.0"),
            "solidity_cut": tk.StringVar(value="0.90"),
            "opening_radius_px": tk.StringVar(value="2"),
            "crop_top_px": tk.StringVar(value="0"),
            "crop_bottom_px": tk.StringVar(value="0"),
        }
        rows = [
            ("픽셀 크기 (µm/px)", "pixel_size_um"),
            ("Gaussian σ (px)", "sigma_px"),
            ("이진화 임계 (0–1)", "threshold"),
            ("최소 등가직경 (µm)", "min_diam_um"),
            ("Solidity 기준", "solidity_cut"),
            ("Opening 반경 (px)", "opening_radius_px"),
            ("상단 크롭 (px)", "crop_top_px"),
            ("하단 크롭 (px)", "crop_bottom_px"),
        ]
        self.param_labels = {}
        for i, (label, key) in enumerate(rows):
            r, c = i % 4, i // 4
            lab = ttk.Label(pbox, text=label, cursor="question_arrow")
            self.param_labels[key] = lab
            lab.grid(row=r, column=c * 2, sticky="w", padx=(0, 6), pady=2)
            ent = ttk.Entry(pbox, textvariable=self.v[key], width=8)
            ent.grid(row=r, column=c * 2 + 1, pady=2)
            help_text = PARAM_HELP.get(key, "")
            if help_text:
                attach_tip(lab, self.tip, help_text, key=f"param:{key}")
                attach_tip(ent, self.tip, help_text, key=f"param:{key}")

        # pixel size helper, right under the parameter grid
        pbtn = ttk.Frame(pbox)
        pbtn.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))
        b_scale = ttk.Button(pbtn, text="스케일바로 측정…", command=self._measure_scale)
        b_scale.pack(side="left")
        attach_tip(b_scale, self.tip,
                   "선택한 이미지를 원본 그대로(정보바 포함) 띄워 드립니다. "
                   "스케일바의 한쪽 끝에서 반대쪽 끝까지 드래그하시고 거기 적힌 "
                   "길이를 입력하시면 픽셀 크기를 계산해 채웁니다.\n\n"
                   "확대경과 수평 고정이 있어 한두 픽셀 오차를 줄일 수 있습니다.",
                   key="btn:scalebar")
        self.lbl_px_src = ttk.Label(pbtn, text="", style="Hint.TLabel")
        self.lbl_px_src.pack(side="left", padx=(8, 0))

        # threshold mode
        mrow = ttk.Frame(pbox)
        mrow.grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))
        lab_mode = ttk.Label(mrow, text="임계 모드", cursor="question_arrow")
        lab_mode.pack(side="left", padx=(0, 6))
        self.var_mode = tk.StringVar(value="고정")
        cmb = ttk.Combobox(mrow, textvariable=self.var_mode, state="readonly",
                           width=15, values=[m[0] for m in THRESHOLD_MODES])
        cmb.pack(side="left")
        attach_tip(lab_mode, self.tip, MODE_HELP, key="mode")
        attach_tip(cmb, self.tip, MODE_HELP, key="mode")

        self.var_normalize = tk.BooleanVar(value=False)
        cb0 = ttk.Checkbutton(mrow, text="대비 정규화", variable=self.var_normalize)
        cb0.pack(side="left", padx=(12, 0))
        attach_tip(cb0, self.tip, CHECK_HELP["normalize"], key="chk:normalize")

        self.var_phys = tk.BooleanVar(value=False)
        cb3 = ttk.Checkbutton(mrow, text="σ·opening을 µm로",
                              variable=self.var_phys, command=self._update_unit_labels)
        cb3.pack(side="left", padx=(12, 0))
        attach_tip(cb3, self.tip, CHECK_HELP["phys"], key="chk:phys")

        b_reco = ttk.Button(mrow, text="권장 설정", command=self._apply_recommended)
        b_reco.pack(side="left", padx=(12, 0))
        attach_tip(b_reco, self.tip,
                   "조건 간 비교에 가장 안전한 조합으로 맞춥니다: "
                   "대비 정규화 켬 + Otsu(일괄).\n\n"
                   "합성 이미지 실험에서 한쪽을 밝고 대비 낮게 찍어도 조건 간 "
                   "비율이 2.24배로 유지되었습니다. 같은 조건에서 고정 임계는 "
                   "1.73배까지 무너집니다.\n\n"
                   "다만 기존 논문 수치(고정 0.45)를 재현하실 때는 '고정'을 "
                   "쓰셔야 합니다.",
                   key="btn:reco")

        self.var_border = tk.BooleanVar(value=True)
        self.var_autocrop = tk.BooleanVar(value=True)
        cb1 = ttk.Checkbutton(pbox, text="프레임 접촉 객체 제외", variable=self.var_border)
        cb1.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
        cb2 = ttk.Checkbutton(pbox, text="위·아래 단색 띠 자동 크롭", variable=self.var_autocrop)
        cb2.grid(row=6, column=2, columnspan=2, sticky="w", pady=(6, 0))
        attach_tip(cb1, self.tip, CHECK_HELP["border"], key="chk:border")
        attach_tip(cb2, self.tip, CHECK_HELP["autocrop"], key="chk:autocrop")

        abox = ttk.Frame(pbox)
        abox.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.btn_run = ttk.Button(abox, text="분석 실행", command=self._run)
        self.btn_run.pack(side="left")
        ttk.Button(abox, text="CSV 저장", command=self._save_csv).pack(side="left", padx=4)
        ttk.Button(abox, text="오버레이 저장", command=self._save_overlay).pack(side="left")
        b_sweep = ttk.Button(abox, text="강건성 스윕", command=self._run_sweep)
        b_sweep.pack(side="left", padx=4)
        attach_tip(b_sweep, self.tip,
                   "선택한 이미지에 대해 이진화 임계 0.35–0.65 × solidity 기준 "
                   "0.85–0.95의 21개 조합으로 개공률을 다시 계산합니다.\n\n"
                   "절대값이 얼마나 움직이는지, 그리고 두 조건을 비교했을 때 "
                   "대소 관계가 뒤집히지 않는지 확인하는 용도입니다.",
                   key="btn:sweep")

        # ---- table + preview -------------------------------------------
        self.lbl_mixed = ttk.Label(outer, text="", style="Warn.TLabel",
                                   anchor="w", justify="left", wraplength=1250)
        self.lbl_mixed.pack(fill="x", pady=(6, 0))

        mid = ttk.Frame(outer)
        mid.pack(fill="both", expand=True, pady=(8, 0))
        # grid, not pack: a Treeview with 12 columns requests ~1000 px and
        # would otherwise squeeze the preview panel down to nothing.
        mid.columnconfigure(0, weight=1, minsize=380)
        mid.columnconfigure(1, weight=0, minsize=PANEL_W)
        mid.rowconfigure(0, weight=1)

        tbox = ttk.LabelFrame(mid, text="결과  (열 제목에 마우스를 올리면 정의와 산출식이 나옵니다)",
                              padding=4)
        tbox.grid(row=0, column=0, sticky="nsew")

        # A fixed-size holder caps what the table asks for; the horizontal
        # scrollbar below covers whatever does not fit.
        holder = tk.Frame(tbox, width=520, height=240)
        holder.pack_propagate(False)
        holder.pack(side="top", fill="both", expand=True)

        self.tree = ttk.Treeview(holder, columns=[c[0] for c in COLS],
                                 show="headings", height=10)
        for key, title, wdt, _desc in COLS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=wdt, anchor="center", stretch=False)
        self.tree.column("file", anchor="w")
        self.tree.pack(side="top", fill="both", expand=True)
        hsb = ttk.Scrollbar(tbox, orient="horizontal", command=self.tree.xview)
        hsb.pack(side="top", fill="x")
        self.tree.config(xscrollcommand=hsb.set)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._show_preview())
        self.tree.bind("<Motion>", self._on_tree_motion, add="+")
        self.tree.bind("<Leave>", self._on_tree_leave, add="+")

        vbox = ttk.LabelFrame(mid, text="분할 결과", padding=6)
        vbox.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        # Pack the text bottom-up FIRST so the canvas can never squeeze it off
        # the window; the canvas then takes whatever vertical space is left.
        self.lbl_sub = ttk.Label(vbox, text="", style="Hint.TLabel", justify="left",
                                 wraplength=PANEL_W - 10)
        self.lbl_sub.pack(side="bottom", anchor="w", fill="x")
        self.lbl_big = ttk.Label(vbox, text="개공률 —", style="Big.TLabel")
        self.lbl_big.pack(side="bottom", anchor="w", pady=(6, 2))

        legend = ttk.Frame(vbox)
        legend.pack(side="bottom", anchor="w", fill="x", pady=(8, 2))
        self._legend_swatch(legend, CONVEX, "볼록 개구부 — 개공률에 포함",
                            "Solidity가 기준값 이상인 객체입니다. 이 객체들의 면적 합이 "
                            "개공률의 분자가 됩니다.")
        self._legend_swatch(legend, CONCAVE, "오목·폐색 — 개공률에서 제외",
                            "Solidity가 기준값 미만인 객체입니다. 무너진 asperity에 가려진 "
                            "개구부나 표면 그림자가 대부분이라 분자에서 뺍니다.\n\n"
                            "필터에서 탈락한 객체(너무 작거나 경계에 닿은 것)는 아예 "
                            "칠해지지 않습니다.")

        # The canvas's requested width is what sets this panel's width. Labels
        # therefore get a FIXED wraplength: deriving it from the panel width
        # would feed back (narrow label -> narrow panel -> narrower label).
        self.canvas = tk.Canvas(vbox, width=PANEL_W, height=300, bg="#1a1a1a",
                                highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)

        # ---- status -----------------------------------------------------
        self.status = ttk.Label(outer, text="대기 중", style="Hint.TLabel", anchor="w")
        self.status.pack(fill="x", pady=(6, 0))

    def _update_unit_labels(self):
        """Switch the σ / opening labels — and their values — between px and µm."""
        px = self._default_px()
        if self.var_phys.get():
            self.param_labels["sigma_px"].config(text="Gaussian σ (µm)")
            self.param_labels["opening_radius_px"].config(text="Opening 반경 (µm)")
            for key, factor in (("sigma_px", px), ("opening_radius_px", px)):
                try:
                    self.v[key].set(f"{float(self.v[key].get()) * factor:.4g}")
                except ValueError:
                    pass
        else:
            self.param_labels["sigma_px"].config(text="Gaussian σ (px)")
            self.param_labels["opening_radius_px"].config(text="Opening 반경 (px)")
            for key in ("sigma_px", "opening_radius_px"):
                try:
                    self.v[key].set(f"{float(self.v[key].get()) / px:.4g}")
                except (ValueError, ZeroDivisionError):
                    pass

    def _mode_key(self) -> str:
        label = self.var_mode.get()
        for name, key in THRESHOLD_MODES:
            if name == label:
                return key
        return "fixed"

    def _apply_recommended(self):
        self.var_normalize.set(True)
        self.var_mode.set("Otsu (일괄)")
        self.status.config(
            text="권장 설정 적용: 대비 정규화 + Otsu(일괄). "
                 "모든 이미지에 같은 임계값이 적용되며, 결과 표의 '적용 임계' 열에서 확인하실 수 있습니다.")

    # ------------------------------------------------- pixel size per image
    def _default_px(self) -> float:
        try:
            return float(self.v["pixel_size_um"].get())
        except ValueError:
            return 0.404

    def _px_for(self, path) -> float:
        """The scale to use for one image: its own if known, else the default."""
        value, _src = self.px_info.get(path, (None, ""))
        return value if value else self._default_px()

    def _refresh_all_rows(self):
        # Labels are computed over the whole batch: two folders often hold the
        # same file names, and "site01.tif" twice tells the user nothing.
        labels = unique_labels(self.files)
        for path in self.files:
            if not self.flist.exists(path):
                continue
            value, src = self.px_info.get(path, (None, ""))
            name = labels.get(path, os.path.basename(path))
            if value:
                self.flist.item(path, values=(name, f"{value:.4g}", src), tags=())
            else:
                self.flist.item(path, values=(name, f"{self._default_px():.4g}",
                                              "기본값 사용"), tags=("unknown",))
        self._check_mixed_scales()

    def _check_mixed_scales(self):
        """Tell the user plainly when the batch mixes magnifications."""
        scales = {round(self._px_for(p), 6) for p in self.files}
        unknown = [p for p in self.files if not self.px_info.get(p, (None,))[0]]
        parts = []
        if len(scales) > 1:
            lo, hi = min(scales), max(scales)
            parts.append(f"배율 혼재: 픽셀 크기가 {lo:.4g}–{hi:.4g} µm/px로 다릅니다"
                         f" ({len(scales)}종). 개공률 자체는 배율에 불변이지만, "
                         f"σ·opening을 µm로 지정하셔야 물리적 기준이 같아집니다.")
        if unknown:
            parts.append(f"{len(unknown)}개 이미지는 픽셀 크기를 모릅니다 — "
                         f"기본값 {self._default_px():.4g}를 씁니다.")
        self.lbl_mixed.config(text="  ".join(parts))
        return len(scales) > 1

    def _edit_pixel_size(self, _event=None):
        paths = list(self.flist.selection()) or (self.files[:1] if self.files else [])
        if not paths:
            messagebox.showinfo("이미지 없음", "먼저 이미지를 추가하고 선택하십시오.")
            return
        _PixelSizeDialog(self, paths)

    def _set_pixel_size(self, paths, value, source):
        for path in paths:
            self.px_info[path] = (value, source)
            self.scale_store.put(path, value, source)
        saved = self.scale_store.save()
        self._refresh_all_rows()
        names = ", ".join(os.path.basename(p) for p in paths[:3])
        more = f" 외 {len(paths)-3}개" if len(paths) > 3 else ""
        self.status.config(
            text=f"{names}{more} 픽셀 크기를 {format_pixel_size(value)}로 "
                 f"설정했습니다. [{source}]"
                 + ("  다음 실행에서도 기억합니다." if saved else
                    "  (저장 실패 — 이번 실행에만 적용됩니다.)"))

    def _measure_scale(self):
        """Scale-bar button in the parameter panel: acts on the selected image."""
        self._edit_pixel_size()

    def _autofill_pixel_size(self, paths):
        """Read µm/px from each new file's own metadata."""
        found = remembered = 0
        for path in paths:
            if path in self.px_info:
                continue
            # A value the user measured by hand outranks the file's metadata:
            # they set it precisely because the metadata was absent or wrong.
            saved = self.scale_store.get(path)
            if saved:
                self.px_info[path] = saved
                remembered += 1
                continue
            got = read_pixel_size(path)
            self.px_info[path] = got if got else (None, "")
            if got:
                found += 1
        self._refresh_all_rows()
        if remembered:
            self.status.config(
                text=f"{remembered}개 이미지는 이전에 지정하신 픽셀 크기를 복원했습니다"
                     + (f", {found}개는 메타데이터에서 읽었습니다." if found else "."))
            return
        if found:
            first = next(p for p in paths if self.px_info.get(p, (None,))[0])
            value, source = self.px_info[first]
            self.lbl_px_src.config(text=f"{found}개 이미지에서 자동 인식")
            self.status.config(
                text=f"{found}개 이미지의 메타데이터에서 픽셀 크기를 읽었습니다 "
                     f"(예: {os.path.basename(first)} = {format_pixel_size(value)}, {source}).")

    def _set_drop_hint(self):
        self.lbl_hint.config(
            text="창에 파일을 끌어다 놓으셔도 됩니다" if self.dnd_ok
            else "이미지 추가 버튼을 사용하십시오")

    def _legend_swatch(self, parent, color, label, help_text):
        row = ttk.Frame(parent)
        row.pack(side="top", anchor="w", fill="x", pady=1)
        sw = tk.Canvas(row, width=14, height=14, highlightthickness=1,
                       highlightbackground="#888", bg=color)
        sw.pack(side="left", padx=(0, 7))
        lab = ttk.Label(row, text=label, style="Legend.TLabel", cursor="question_arrow")
        lab.pack(side="left")
        attach_tip(lab, self.tip, help_text, key=f"legend:{color}")
        attach_tip(sw, self.tip, help_text, key=f"legend:{color}")

    def _on_tree_leave(self, _event):
        """
        Mapping the tooltip window makes X deliver a <Leave> to the table, so
        hiding straight away would erase the tooltip the moment it appears.
        Re-check where the pointer actually is a moment later instead.
        """
        self.root.after(150, self._hide_tip_if_pointer_left)

    def _hide_tip_if_pointer_left(self):
        try:
            w = self.root.winfo_containing(self.root.winfo_pointerx(),
                                           self.root.winfo_pointery())
        except Exception:
            w = None
        if w is self.tree:
            return
        win = self.tip._win
        if win is not None and w is not None and str(w).startswith(str(win)):
            return                        # pointer is over the tooltip itself
        self.tip.hide()

    def _on_tree_motion(self, event):
        if self.tree.identify_region(event.x, event.y) != "heading":
            self.tip.hide()
            return
        col = self.tree.identify_column(event.x)          # "#1", "#2", ...
        try:
            i = int(col.lstrip("#")) - 1
        except ValueError:
            return
        if not (0 <= i < len(COLS)):
            return
        key, title, _w, desc = COLS[i]
        self.tip.request(f"col:{key}", f"{title}\n\n{desc}",
                         event.x_root + 12, event.y_root + 20)

    # ------------------------------------------------------------- files
    def _browse(self):
        paths = filedialog.askopenfilenames(
            title="SEM 이미지 선택",
            filetypes=[("이미지", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("모든 파일", "*.*")])
        self._add_files(paths)

    def _on_drop(self, event):
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        self._add_files(paths)

    def _add_files(self, paths):
        new = []
        for p in paths:
            p = str(p).strip("{}")
            if os.path.isdir(p):
                for fn in sorted(os.listdir(p)):
                    fp = os.path.join(p, fn)
                    if fp.lower().endswith(IMAGE_EXT) and fp not in self.files:
                        new.append(fp)
            elif p.lower().endswith(IMAGE_EXT) and p not in self.files:
                new.append(p)
        for fp in new:
            self.files.append(fp)
            self.flist.insert("", "end", iid=fp,
                              values=(os.path.basename(fp), "", ""))
        if new:
            self.status.config(text=f"{len(new)}개 추가 — 총 {len(self.files)}개")
            self._autofill_pixel_size(new)

    def _remove_sel(self):
        for path in list(self.flist.selection()):
            self.results.pop(path, None)
            self.px_info.pop(path, None)
            if path in self.files:
                self.files.remove(path)
            self.flist.delete(path)
        self._refresh_table()
        self._check_mixed_scales()

    def _clear(self):
        self.files.clear()
        self.results.clear()
        self.px_info.clear()
        self.flist.delete(*self.flist.get_children())
        self._refresh_table()
        self.canvas.delete("all")
        self.lbl_big.config(text="개공률 —")
        self.lbl_sub.config(text="")
        self.lbl_mixed.config(text="")
        self.lbl_px_src.config(text="")

    # ------------------------------------------------------------ params
    def _params(self) -> Params:
        def f(key, cast=float):
            try:
                return cast(self.v[key].get())
            except ValueError:
                raise ValueError(f"'{key}' 값이 숫자가 아닙니다: {self.v[key].get()}")
        phys = self.var_phys.get()
        p = Params(
            pixel_size_um=f("pixel_size_um"),
            sigma_px=1.2 if phys else f("sigma_px"),
            sigma_um=f("sigma_px") if phys else None,
            threshold=f("threshold"),
            min_diam_um=f("min_diam_um"),
            solidity_cut=f("solidity_cut"),
            opening_radius_px=2 if phys else f("opening_radius_px", int),
            opening_radius_um=f("opening_radius_px") if phys else None,
            crop_top_px=f("crop_top_px", int),
            crop_bottom_px=f("crop_bottom_px", int),
            exclude_border=self.var_border.get(),
            normalize_contrast=self.var_normalize.get(),
            threshold_mode=self._mode_key(),
        )
        if p.pixel_size_um <= 0:
            raise ValueError("픽셀 크기는 0보다 커야 합니다.")
        if not (0.0 < p.threshold < 1.0):
            raise ValueError("이진화 임계는 0과 1 사이여야 합니다.")
        return p

    # --------------------------------------------------------------- run
    def _run(self):
        if self._busy:
            return
        if not self.files:
            messagebox.showinfo("이미지 없음", "분석할 이미지를 먼저 추가하십시오.")
            return
        try:
            p = self._params()
        except ValueError as e:
            messagebox.showerror("입력 오류", str(e))
            return
        self._busy = True
        self._failures = []
        self.btn_run.config(state="disabled")
        # Read every Tk variable here, on the main thread. Tkinter is not
        # thread-safe: touching a Tk variable from the worker deadlocks Tcl.
        autocrop = self.var_autocrop.get()
        px_map = {path: self._px_for(path) for path in self.files}
        labels = unique_labels(self.files)
        threading.Thread(target=self._worker,
                         args=(list(self.files), p, autocrop, px_map, labels),
                         daemon=True).start()

    def _worker(self, files, p, autocrop, px_map, labels):
        # Pass 1: settle each file's own scale and crop. The batch threshold has
        # to be computed from exactly the pixels that will be analysed, so
        # cropping must be decided before the histogram is pooled.
        plans = []
        for path in files:
            pp = p.copy_with(pixel_size_um=px_map.get(path, p.pixel_size_um))
            if autocrop and p.crop_top_px == 0 and p.crop_bottom_px == 0:
                try:
                    top, bot = detect_bands(load_gray(path))
                except Exception:
                    top = bot = 0
                if top or bot:
                    pp = pp.copy_with(crop_top_px=top, crop_bottom_px=bot)
            plans.append((path, pp))

        # Pass 2: one threshold from all images, when that mode is selected.
        # A generator keeps only one image in memory at a time.
        forced = None
        if p.threshold_mode == "otsu_batch":
            self._q.put(("status", "모든 이미지의 밝기 분포를 합쳐 공통 임계값 계산 중…"))

            def grays():
                for path, pp in plans:
                    try:
                        yield load_gray(path, pp.crop_bottom_px, pp.crop_top_px)
                    except Exception:
                        continue
            forced = pooled_otsu(grays(), p)
            if forced == forced:                      # not NaN
                self._q.put(("status",
                             f"공통 임계값 {forced:.4f} — 모든 이미지에 동일하게 적용합니다."))
            else:
                forced = None
                self._q.put(("status",
                             "공통 임계값을 구하지 못해 입력하신 고정 임계값을 사용합니다."))

        # Pass 3: analyse
        for i, (path, pp) in enumerate(plans, 1):
            try:
                self._q.put(("status",
                             f"[{i}/{len(plans)}] {labels.get(path, os.path.basename(path))} "
                             f"분석 중…"))
                gray = load_gray(path, pp.crop_bottom_px, pp.crop_top_px)
                res = analyze_array(gray, pp,
                                    source=labels.get(path, os.path.basename(path)),
                                    forced_threshold=forced)
                res.source_path = path
                # Keep only what the preview needs. A full-resolution overlay is
                # ~9 MiB for a 2048x1536 SEM image, so a 100-image batch would
                # sit on close to a gigabyte of pictures drawn at 460 px wide.
                # The full one is regenerated on demand when saving.
                res.binary = None
                res.overlay = _shrink_overlay(res.overlay)
                self._q.put(("result", (path, res, pp)))
            except Exception:
                self._q.put(("error", (path, traceback.format_exc(limit=3))))
        self._q.put(("done", None))

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "status":
                    self.status.config(text=payload)
                elif kind == "result":
                    path, res, used = payload
                    res.used_params = used
                    self.results[path] = res
                    self._refresh_table()
                elif kind == "error":
                    # Collect, never interrupt. One modal per bad file would
                    # stall a batch behind a dialog nobody is there to close.
                    path, tb = payload
                    self._failures.append((path, tb))
                    self.status.config(
                        text=f"실패 {len(self._failures)}건 — {os.path.basename(path)} "
                             f"(계속 진행합니다)")
                elif kind == "done":
                    self._busy = False
                    self.btn_run.config(state="normal")
                    self._report_failures()
                    self._show_preview()
        except queue.Empty:
            pass
        self.root.after(80, self._drain_queue)

    def _report_failures(self):
        """One summary at the end, rather than a dialog per failed file."""
        ok = len(self.results)
        if not self._failures:
            self.status.config(text=f"완료 — {ok}개 결과")
            return
        n = len(self._failures)
        self.status.config(text=f"완료 — 성공 {ok}개, 실패 {n}개")
        lines = []
        for path, tb in self._failures[:10]:
            last = [ln for ln in tb.strip().splitlines() if ln.strip()][-1]
            lines.append(f"• {os.path.basename(path)}\n    {last.strip()[:110]}")
        more = f"\n… 외 {n - 10}건" if n > 10 else ""
        messagebox.showwarning(
            "일부 파일 분석 실패",
            f"{ok}개는 정상 분석되었고 {n}개가 실패했습니다.\n"
            f"실패한 파일은 결과와 CSV에서 제외됩니다.\n\n"
            + "\n".join(lines) + more)

    def _run_sweep(self):
        sel = self._selected_path()
        if sel is None or sel not in self.results:
            messagebox.showinfo("선택 필요", "먼저 분석을 실행하고 이미지를 선택하십시오.")
            return
        try:
            p = self._params()
        except ValueError as e:
            messagebox.showerror("입력 오류", str(e))
            return
        used = getattr(self.results[sel], "used_params", p)
        self.status.config(text="강건성 스윕 계산 중…")
        self.root.update_idletasks()
        gray = load_gray(sel, used.crop_bottom_px, used.crop_top_px)
        rows = robustness_sweep(gray, used)
        self.status.config(text="스윕 완료")

        win = tk.Toplevel(self.root)
        win.title(f"강건성 스윕 — {os.path.basename(sel)}")
        win.geometry("540x470")
        ttk.Label(win, padding=8, justify="left", wraplength=510,
                  text=("임계 0.35–0.65 × solidity 0.85–0.95 조합별 개공률입니다. "
                        "절대값은 크게 변하므로 두 조건을 동일 설정에서 비교한 "
                        "대소 관계만 사용하십시오.")).pack(fill="x")
        tv = ttk.Treeview(win, columns=("t", "s", "f", "n"), show="headings")
        for k, t, w in (("t", "임계", 95), ("s", "Solidity", 95),
                        ("f", "개공률 %", 115), ("n", "객체수", 95)):
            tv.heading(k, text=t)
            tv.column(k, width=w, anchor="center")
        tv.pack(fill="both", expand=True, padx=8, pady=8)
        for r in rows:
            tv.insert("", "end", values=(r["threshold"], r["solidity_cut"],
                                         r["open_pore_fraction_pct"], r["n_objects"]))

        def save():
            fp = filedialog.asksaveasfilename(defaultextension=".csv",
                                              filetypes=[("CSV", "*.csv")],
                                              initialfile="robustness_sweep.csv")
            if fp:
                with open(fp, "w", newline="", encoding="utf-8-sig") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                messagebox.showinfo("저장", f"저장했습니다:\n{fp}")
        ttk.Button(win, text="CSV 저장", command=save).pack(pady=(0, 8))

    # ------------------------------------------------------------- table
    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for path in self.files:
            res = self.results.get(path)
            if res is None:
                continue
            row = res.summary_row()
            self.tree.insert("", "end", iid=path,
                             values=[row.get(c[0], "") for c in COLS])

    def _selected_path(self):
        sel = self.tree.selection()
        if sel:
            return sel[0]
        sel = self.flist.selection()
        if sel:
            return sel[0]
        for path in self.files:
            if path in self.results:
                return path
        return None

    # ----------------------------------------------------------- preview
    def _show_preview(self):
        path = self._selected_path()
        if path is None or path not in self.results:
            return
        res = self.results[path]
        ov = res.overlay
        if ov is None:
            return
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 140)
        im = Image.fromarray(ov)
        scale = min(cw / im.width, ch / im.height)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.LANCZOS)
        self._preview_img = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._preview_img, anchor="center")

        used = getattr(res, "used_params", None)
        crop_note = ""
        if used is not None and (used.crop_top_px or used.crop_bottom_px):
            crop_note = (f"자동 크롭 위 {used.crop_top_px} px / 아래 "
                         f"{used.crop_bottom_px} px 적용 후 측정\n")

        mode_note = ""
        if used is not None:
            label = {"fixed": "고정", "otsu": "Otsu(이미지별)",
                     "otsu_batch": "Otsu(일괄)"}.get(used.threshold_mode, used.threshold_mode)
            norm = " · 대비 정규화" if used.normalize_contrast else ""
            mode_note = (f"임계 {res.effective_threshold:.4f} "
                         f"({label}{norm}) · 이 이미지의 Otsu {res.otsu_threshold:.3f}\n")

        self.lbl_big.config(text=f"개공률 {100*res.open_pore_fraction:.2f} %")
        self.lbl_sub.config(text=(
            f"객체 {res.n_objects}개 · 원형도 중앙값 {res.circularity_median:.3f} · "
            f"Solidity 중앙값 {res.solidity_median:.3f}\n"
            f"시야 {res.field_w_um:.0f} × {res.field_h_um:.0f} µm "
            f"({res.width_px} × {res.height_px} px)\n"
            f"{mode_note}"
            f"{crop_note}"
            f"단순 암부면적률 {100*res.dark_area_fraction:.1f} % — 그림자 포함, 참고용\n"
            f"프레임 접촉 제외 {res.n_rejected_border}개 "
            f"(면적 {100*res.border_area_fraction:.1f} %) — 개공률은 그만큼 과소평가"))

    # -------------------------------------------------------------- save
    def _save_csv(self):
        if not self.results:
            messagebox.showinfo("결과 없음", "먼저 분석을 실행하십시오.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".csv",
                                          filetypes=[("CSV", "*.csv")],
                                          initialfile="pore_analysis.csv")
        if not fp:
            return
        rows = [self.results[p].summary_row() for p in self.files if p in self.results]
        with open(fp, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        self.status.config(text=f"CSV 저장: {fp}")
        messagebox.showinfo("저장 완료", fp)

    def _save_overlay(self):
        path = self._selected_path()
        if path is None or path not in self.results:
            messagebox.showinfo("선택 필요", "저장할 결과를 선택하십시오.")
            return
        base = os.path.splitext(os.path.basename(path))[0]
        fp = filedialog.asksaveasfilename(defaultextension=".png",
                                          filetypes=[("PNG", "*.png")],
                                          initialfile=f"{base}_overlay.png")
        if not fp:
            return
        res = self.results[path]
        used = getattr(res, "used_params", None)
        overlay = res.overlay
        if used is not None:
            # Only a preview-sized copy is kept in memory; redo this one image
            # so the saved file is at the original resolution.
            self.status.config(text="원본 해상도로 오버레이를 다시 그리는 중…")
            self.root.update_idletasks()
            try:
                gray = load_gray(path, used.crop_bottom_px, used.crop_top_px)
                full = analyze_array(gray, used, source=res.source,
                                     forced_threshold=res.effective_threshold)
                overlay = full.overlay
            except Exception:
                pass                       # fall back to the preview-sized one
        Image.fromarray(overlay).save(fp)
        self.status.config(text=f"오버레이 저장: {fp}  ({overlay.shape[1]}×{overlay.shape[0]} px)")


class _PixelSizeDialog:
    """Set µm/px for the selected images: type it, or measure the scale bar."""

    def __init__(self, app, paths):
        self.app = app
        self.paths = paths
        first = paths[0]

        win = tk.Toplevel(app.root)
        self.win = win
        win.title("픽셀 크기 지정")
        win.transient(app.root)
        win.resizable(False, False)

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)

        names = ", ".join(os.path.basename(p) for p in paths[:3])
        more = f" 외 {len(paths)-3}개" if len(paths) > 3 else ""
        ttk.Label(frm, text=f"대상: {names}{more}", wraplength=380,
                  justify="left").pack(anchor="w")

        cur, src = app.px_info.get(first, (None, ""))
        ttk.Label(frm, style="Hint.TLabel", wraplength=380, justify="left",
                  text=(f"현재 값: {format_pixel_size(cur)}  [{src}]" if cur
                        else "현재 값 없음 — 분석 시 기본값이 쓰입니다.")
                  ).pack(anchor="w", pady=(2, 10))

        row = ttk.Frame(frm)
        row.pack(anchor="w")
        ttk.Label(row, text="µm/px").pack(side="left", padx=(0, 6))
        self.var = tk.StringVar(value=f"{cur:.6g}" if cur else
                                f"{app._default_px():.6g}")
        ttk.Entry(row, textvariable=self.var, width=14).pack(side="left")

        ttk.Button(frm, text="스케일바로 측정…", command=self._measure
                   ).pack(anchor="w", pady=(10, 0))
        ttk.Label(frm, style="Hint.TLabel", wraplength=380, justify="left",
                  text="첫 번째 이미지를 원본 그대로 띄워 스케일바를 재고, "
                       "그 값을 위 칸에 넣습니다."
                  ).pack(anchor="w")

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(14, 0))
        ttk.Button(btns, text="적용", command=self._apply).pack(side="right")
        ttk.Button(btns, text="취소", command=win.destroy).pack(side="right", padx=6)
        if len(app.files) > len(paths):
            ttk.Button(btns, text="전체 이미지에 적용",
                       command=self._apply_all).pack(side="left")

        win.bind("<Return>", lambda e: self._apply())
        win.bind("<Escape>", lambda e: win.destroy())
        win.update_idletasks()
        win.grab_set()

    def _measure(self):
        try:
            value = measure_scale_bar(self.win, self.paths[0])
        except Exception:
            messagebox.showerror("측정 실패", traceback.format_exc(limit=3),
                                 parent=self.win)
            return
        if value:
            self.var.set(f"{value:.6g}")
            self._source = "스케일바 측정"

    def _value(self):
        try:
            v = float(self.var.get())
        except ValueError:
            messagebox.showerror("입력 오류", "숫자를 입력하십시오.", parent=self.win)
            return None
        if v <= 0:
            messagebox.showerror("입력 오류", "픽셀 크기는 0보다 커야 합니다.",
                                 parent=self.win)
            return None
        return v

    def _apply(self, targets=None):
        v = self._value()
        if v is None:
            return
        self.app._set_pixel_size(targets or self.paths, v,
                                 getattr(self, "_source", "직접 입력"))
        self.win.destroy()

    def _apply_all(self):
        self._apply(targets=list(self.app.files))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    App(root, initial_files=args)
    root.mainloop()


if __name__ == "__main__":
    main()
