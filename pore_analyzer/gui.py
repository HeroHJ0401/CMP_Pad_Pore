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

from .core import (Params, analyze_path, load_gray, analyze_array,
                   robustness_sweep, detect_bands)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:
    HAS_DND = False

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

CONVEX = "#2ecc71"      # solidity >= cut  -> counted in the open-pore fraction
CONCAVE = "#e74c3c"     # solidity <  cut  -> excluded
PANEL_W = 460           # preview panel width; also the label wrap width

# key, heading, width, tooltip
COLS = [
    ("file", "파일", 170,
     "분석한 이미지 파일명입니다."),
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
    ("otsu_threshold_ref", "Otsu(참고)", 84,
     "이 이미지에서 자동으로 산출된 Otsu 임계값입니다. 계산에는 쓰이지 "
     "않습니다.\n\n"
     "설정하신 이진화 임계값이 이 값과 크게 다르면, 조명이나 대비가 다른 "
     "이미지를 같은 고정 임계로 비교하고 있지 않은지 확인해 보십시오."),
]

PARAM_HELP = {
    "pixel_size_um":
        "이미지 한 픽셀이 실제로 몇 µm인지입니다. 배율마다 다르므로 SEM 스케일바로 "
        "확인하십시오.\n\n면적·등가직경·밀도·개공률이 모두 이 값에 의존합니다. "
        "잘못 넣으면 형상 지표(원형도·solidity)는 그대로지만 크기 관련 값이 전부 "
        "틀어집니다.",
    "sigma_px":
        "이진화 전에 적용하는 가우시안 평활의 표준편차(픽셀)입니다.\n\n"
        "각 픽셀은 자기 밝기가 아니라 주변 약 ±3σ 이웃의 가중평균과 임계값을 "
        "비교하게 됩니다. 픽셀 노이즈가 작은 점 객체로 잡히는 것을 막습니다.\n\n"
        "0을 넣으면 평활을 건너뜁니다.",
    "threshold":
        "정규화 밝기(0~1) 기준으로 이 값보다 어두운 픽셀을 개구부 후보로 봅니다.\n\n"
        "고정 임계값이므로 두 조건을 비교하실 때는 반드시 같은 값을 쓰셔야 합니다. "
        "값 자체의 타당성은 오른쪽 표의 Otsu 참고값과 대조해 보십시오.",
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

CHECK_HELP = {
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
        root.title("CMP Pad 개공률 분석기")
        root.geometry("1320x840")
        root.minsize(1100, 700)

        self.files: list[str] = []
        self.results: dict[str, object] = {}
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

    def _build_widgets(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")

        # ---- file list -------------------------------------------------
        fbox = ttk.LabelFrame(top, text="이미지", padding=6)
        fbox.pack(side="left", fill="both", expand=True)

        self.lst = tk.Listbox(fbox, height=6, activestyle="dotbox",
                              selectmode="extended", exportselection=False)
        self.lst.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(fbox, orient="vertical", command=self.lst.yview)
        sb.pack(side="left", fill="y")
        self.lst.config(yscrollcommand=sb.set)
        self.lst.bind("<<ListboxSelect>>", lambda e: self._show_preview())

        fbtn = ttk.Frame(fbox)
        fbtn.pack(side="left", fill="y", padx=(8, 0))
        ttk.Button(fbtn, text="이미지 추가", command=self._browse).pack(fill="x", pady=2)
        ttk.Button(fbtn, text="선택 제거", command=self._remove_sel).pack(fill="x", pady=2)
        ttk.Button(fbtn, text="전체 비우기", command=self._clear).pack(fill="x", pady=2)
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
        for i, (label, key) in enumerate(rows):
            r, c = i % 4, i // 4
            lab = ttk.Label(pbox, text=label, cursor="question_arrow")
            lab.grid(row=r, column=c * 2, sticky="w", padx=(0, 6), pady=2)
            ent = ttk.Entry(pbox, textvariable=self.v[key], width=8)
            ent.grid(row=r, column=c * 2 + 1, pady=2)
            help_text = PARAM_HELP.get(key, "")
            if help_text:
                attach_tip(lab, self.tip, help_text, key=f"param:{key}")
                attach_tip(ent, self.tip, help_text, key=f"param:{key}")

        self.var_border = tk.BooleanVar(value=True)
        self.var_autocrop = tk.BooleanVar(value=True)
        cb1 = ttk.Checkbutton(pbox, text="프레임 접촉 객체 제외", variable=self.var_border)
        cb1.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        cb2 = ttk.Checkbutton(pbox, text="위·아래 단색 띠 자동 크롭", variable=self.var_autocrop)
        cb2.grid(row=4, column=2, columnspan=2, sticky="w", pady=(6, 0))
        attach_tip(cb1, self.tip, CHECK_HELP["border"], key="chk:border")
        attach_tip(cb2, self.tip, CHECK_HELP["autocrop"], key="chk:autocrop")

        abox = ttk.Frame(pbox)
        abox.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
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
        added = 0
        for p in paths:
            p = str(p).strip("{}")
            if os.path.isdir(p):
                for fn in sorted(os.listdir(p)):
                    fp = os.path.join(p, fn)
                    if fp.lower().endswith(IMAGE_EXT) and fp not in self.files:
                        self.files.append(fp)
                        self.lst.insert("end", os.path.basename(fp))
                        added += 1
            elif p.lower().endswith(IMAGE_EXT) and p not in self.files:
                self.files.append(p)
                self.lst.insert("end", os.path.basename(p))
                added += 1
        if added:
            self.status.config(text=f"{added}개 추가 — 총 {len(self.files)}개")

    def _remove_sel(self):
        for i in sorted(self.lst.curselection(), reverse=True):
            self.results.pop(self.files[i], None)
            del self.files[i]
            self.lst.delete(i)
        self._refresh_table()

    def _clear(self):
        self.files.clear()
        self.results.clear()
        self.lst.delete(0, "end")
        self._refresh_table()
        self.canvas.delete("all")
        self.lbl_big.config(text="개공률 —")
        self.lbl_sub.config(text="")

    # ------------------------------------------------------------ params
    def _params(self) -> Params:
        def f(key, cast=float):
            try:
                return cast(self.v[key].get())
            except ValueError:
                raise ValueError(f"'{key}' 값이 숫자가 아닙니다: {self.v[key].get()}")
        p = Params(
            pixel_size_um=f("pixel_size_um"),
            sigma_px=f("sigma_px"),
            threshold=f("threshold"),
            min_diam_um=f("min_diam_um"),
            solidity_cut=f("solidity_cut"),
            opening_radius_px=f("opening_radius_px", int),
            crop_top_px=f("crop_top_px", int),
            crop_bottom_px=f("crop_bottom_px", int),
            exclude_border=self.var_border.get(),
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
        self.btn_run.config(state="disabled")
        # Read every Tk variable here, on the main thread. Tkinter is not
        # thread-safe: touching a Tk variable from the worker deadlocks Tcl.
        autocrop = self.var_autocrop.get()
        threading.Thread(target=self._worker,
                         args=(list(self.files), p, autocrop), daemon=True).start()

    def _worker(self, files, p, autocrop):
        for i, path in enumerate(files, 1):
            try:
                self._q.put(("status", f"[{i}/{len(files)}] {os.path.basename(path)} 분석 중…"))
                pp = p
                if autocrop and p.crop_top_px == 0 and p.crop_bottom_px == 0:
                    top, bot = detect_bands(load_gray(path))
                    if top or bot:
                        pp = p.copy_with(crop_top_px=top, crop_bottom_px=bot)
                        self._q.put(("status",
                                     f"{os.path.basename(path)}: 단색 띠 자동 크롭 "
                                     f"(위 {top} px, 아래 {bot} px)"))
                res = analyze_path(path, pp)
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
                    path, tb = payload
                    self.status.config(text=f"실패: {os.path.basename(path)}")
                    messagebox.showerror("분석 실패", f"{os.path.basename(path)}\n\n{tb}")
                elif kind == "done":
                    self._busy = False
                    self.btn_run.config(state="normal")
                    self.status.config(text=f"완료 — {len(self.results)}개 결과")
                    self._show_preview()
        except queue.Empty:
            pass
        self.root.after(80, self._drain_queue)

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
        idx = self.lst.curselection()
        if idx:
            return self.files[idx[0]]
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

        self.lbl_big.config(text=f"개공률 {100*res.open_pore_fraction:.2f} %")
        self.lbl_sub.config(text=(
            f"객체 {res.n_objects}개 · 원형도 중앙값 {res.circularity_median:.3f} · "
            f"Solidity 중앙값 {res.solidity_median:.3f}\n"
            f"시야 {res.field_w_um:.0f} × {res.field_h_um:.0f} µm "
            f"({res.width_px} × {res.height_px} px)\n"
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
        Image.fromarray(self.results[path].overlay).save(fp)
        self.status.config(text=f"오버레이 저장: {fp}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    App(root, initial_files=args)
    root.mainloop()


if __name__ == "__main__":
    main()
