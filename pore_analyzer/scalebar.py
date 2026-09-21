"""
Measure µm/px by dragging along the scale bar printed on the image.

The dialog deliberately shows the WHOLE image, info bar included: the scale bar
usually lives in exactly the strip the analysis crops away.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from PIL import Image, ImageTk

from .pixelsize import from_scale_bar, read_pixel_size, format_pixel_size

__all__ = ["measure_scale_bar"]

VIEW_W, VIEW_H = 940, 560
MAG_SIZE = 130          # magnifier viewport, screen px
MAG_ZOOM = 7            # magnification factor
UNITS = ["µm", "nm", "mm"]


class _Dialog:
    def __init__(self, parent, image_path: str):
        self.result: Optional[float] = None
        self.path = image_path
        self.p1 = None          # (x, y) in SOURCE pixels
        self.p2 = None
        self._dragging = False

        self.img = Image.open(image_path)
        if self.img.mode not in ("L", "RGB"):
            self.img = self.img.convert("RGB")
        self.src_w, self.src_h = self.img.size
        self.scale = min(VIEW_W / self.src_w, VIEW_H / self.src_h, 1.0)
        disp = self.img.resize((max(1, int(self.src_w * self.scale)),
                                max(1, int(self.src_h * self.scale))),
                               Image.LANCZOS)

        self.win = tk.Toplevel(parent)
        self.win.title("스케일바로 픽셀 크기 측정")
        self.win.transient(parent)

        self._build(disp)
        self._prefill_from_metadata()

        self.win.update_idletasks()
        self.win.grab_set()

    # ---------------------------------------------------------------- ui
    def _build(self, disp):
        root = ttk.Frame(self.win, padding=10)
        root.pack(fill="both", expand=True)

        ttk.Label(root, justify="left", wraplength=VIEW_W, text=(
            "이미지에 인쇄된 스케일바의 한쪽 끝에서 반대쪽 끝까지 드래그하신 뒤, "
            "바 옆에 적힌 길이를 입력하십시오. 정보바를 포함한 원본 전체를 "
            "보여드리므로 하단의 스케일바도 그대로 보입니다.")
        ).pack(anchor="w", pady=(0, 8))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(body, width=disp.width, height=disp.height,
                                bg="#111", highlightthickness=1,
                                highlightbackground="#555", cursor="crosshair")
        self.canvas.pack(side="left")
        self._disp_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, image=self._disp_img, anchor="nw")

        side = ttk.Frame(body, padding=(12, 0, 0, 0))
        side.pack(side="left", fill="y")

        ttk.Label(side, text="확대경", style="Hint.TLabel").pack(anchor="w")
        self.mag = tk.Canvas(side, width=MAG_SIZE, height=MAG_SIZE, bg="#111",
                             highlightthickness=1, highlightbackground="#555")
        self.mag.pack(pady=(2, 10))

        self.var_horiz = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="수평으로 고정", variable=self.var_horiz,
                        command=self._redraw).pack(anchor="w")
        ttk.Label(side, style="Hint.TLabel", wraplength=180, justify="left",
                  text="스케일바는 거의 항상 수평입니다. 켜 두시면 위아래로 "
                       "흔들려도 길이가 달라지지 않습니다.").pack(anchor="w", pady=(0, 10))

        self.lbl_len = ttk.Label(side, text="바 길이: —")
        self.lbl_len.pack(anchor="w")

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(side, text="바에 적힌 길이").pack(anchor="w")
        row = ttk.Frame(side)
        row.pack(anchor="w", pady=(2, 8))
        self.var_len = tk.StringVar(value="100")
        e = ttk.Entry(row, textvariable=self.var_len, width=9)
        e.pack(side="left")
        self.var_unit = tk.StringVar(value="µm")
        ttk.Combobox(row, textvariable=self.var_unit, values=UNITS,
                     width=5, state="readonly").pack(side="left", padx=4)
        self.var_len.trace_add("write", lambda *_: self._recompute())
        self.var_unit.trace_add("write", lambda *_: self._recompute())

        self.lbl_result = ttk.Label(side, text="픽셀 크기: —",
                                    font=("Segoe UI", 11, "bold"))
        self.lbl_result.pack(anchor="w", pady=(4, 2))
        self.lbl_note = ttk.Label(side, text="", style="Hint.TLabel",
                                  wraplength=180, justify="left")
        self.lbl_note.pack(anchor="w")

        btns = ttk.Frame(root)
        btns.pack(fill="x", pady=(10, 0))
        self.btn_ok = ttk.Button(btns, text="적용", command=self._apply, state="disabled")
        self.btn_ok.pack(side="right")
        ttk.Button(btns, text="취소", command=self._cancel).pack(side="right", padx=6)
        self.lbl_meta = ttk.Label(btns, text="", style="Hint.TLabel")
        self.lbl_meta.pack(side="left")

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_move)
        self.canvas.bind("<Leave>", lambda e: self.mag.delete("all"))
        self.win.bind("<Escape>", lambda e: self._cancel())

    def _prefill_from_metadata(self):
        got = read_pixel_size(self.path)
        if got:
            self.lbl_meta.config(
                text=f"이 파일의 메타데이터: {format_pixel_size(got[0])}  [{got[1]}]")
            self._meta_value = got[0]
        else:
            self.lbl_meta.config(text="이 파일에는 픽셀 크기 메타데이터가 없습니다.")
            self._meta_value = None

    # ------------------------------------------------------------ events
    def _to_src(self, x, y):
        return x / self.scale, y / self.scale

    def _to_disp(self, x, y):
        return x * self.scale, y * self.scale

    def _on_press(self, e):
        self._dragging = True
        self.p1 = self._to_src(e.x, e.y)
        self.p2 = self.p1
        self._redraw()

    def _on_drag(self, e):
        if not self._dragging:
            return
        self.p2 = self._to_src(e.x, e.y)
        self._redraw()
        self._draw_magnifier(e.x, e.y)

    def _on_release(self, e):
        if not self._dragging:
            return
        self._dragging = False
        self.p2 = self._to_src(e.x, e.y)
        self._redraw()

    def _on_move(self, e):
        if not self._dragging:
            self._draw_magnifier(e.x, e.y)

    def _draw_magnifier(self, dx, dy):
        sx, sy = self._to_src(dx, dy)
        half = MAG_SIZE / (2 * MAG_ZOOM)
        box = (int(sx - half), int(sy - half), int(sx + half), int(sy + half))
        try:
            crop = self.img.crop(box).resize((MAG_SIZE, MAG_SIZE), Image.NEAREST)
        except Exception:
            return
        self._mag_img = ImageTk.PhotoImage(crop)
        self.mag.delete("all")
        self.mag.create_image(0, 0, image=self._mag_img, anchor="nw")
        c = MAG_SIZE // 2
        self.mag.create_line(c, 0, c, MAG_SIZE, fill="#ff3b30")
        self.mag.create_line(0, c, MAG_SIZE, c, fill="#ff3b30")

    # --------------------------------------------------------- geometry
    def _endpoints(self):
        """The two endpoints in source pixels, honouring the horizontal lock."""
        if self.p1 is None or self.p2 is None:
            return None
        x1, y1 = self.p1
        x2, y2 = self.p2
        if self.var_horiz.get():
            y2 = y1
        return (x1, y1), (x2, y2)

    def _length_px(self):
        pts = self._endpoints()
        if not pts:
            return 0.0
        (x1, y1), (x2, y2) = pts
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _redraw(self):
        self.canvas.delete("measure")
        pts = self._endpoints()
        if not pts:
            return
        (x1, y1), (x2, y2) = pts
        dx1, dy1 = self._to_disp(x1, y1)
        dx2, dy2 = self._to_disp(x2, y2)
        self.canvas.create_line(dx1, dy1, dx2, dy2, fill="#ffcc00", width=2,
                                tags="measure")
        for dx, dy in ((dx1, dy1), (dx2, dy2)):      # end ticks
            self.canvas.create_line(dx, dy - 9, dx, dy + 9, fill="#ffcc00",
                                    width=2, tags="measure")
        self.lbl_len.config(text=f"바 길이: {self._length_px():.1f} px")
        self._recompute()

    def _recompute(self):
        length_px = self._length_px()
        if length_px <= 0:
            self.lbl_result.config(text="픽셀 크기: —")
            self.lbl_note.config(text="")
            self.btn_ok.config(state="disabled")
            return
        try:
            value = float(self.var_len.get())
        except ValueError:
            self.lbl_result.config(text="픽셀 크기: —")
            self.lbl_note.config(text="바에 적힌 길이를 숫자로 입력하십시오.")
            self.btn_ok.config(state="disabled")
            return
        try:
            um_px = from_scale_bar(length_px, value, self.var_unit.get())
        except ValueError as err:
            self.lbl_result.config(text="픽셀 크기: —")
            self.lbl_note.config(text=str(err))
            self.btn_ok.config(state="disabled")
            return

        self._value = um_px
        self.lbl_result.config(text=format_pixel_size(um_px))
        note = f"시야 전체: {self.src_w * um_px:.0f} × {self.src_h * um_px:.0f} µm"
        if self._meta_value:
            diff = abs(um_px - self._meta_value) / self._meta_value * 100
            note += f"\n메타데이터와 {diff:.1f} % 차이"
            if diff > 10:
                note += " — 한쪽이 틀렸을 수 있으니 확인하십시오."
        self.lbl_note.config(text=note)
        self.btn_ok.config(state="normal")

    # ----------------------------------------------------------- finish
    def _apply(self):
        length_px = self._length_px()
        if length_px < 20:
            if not messagebox.askyesno(
                    "짧은 측정",
                    f"지정하신 길이가 {length_px:.0f} 픽셀뿐입니다.\n\n"
                    "짧을수록 한두 픽셀의 오차가 그대로 비율 오차가 됩니다. "
                    "가능하면 스케일바 전체를 끝에서 끝까지 지정하십시오.\n\n"
                    "이대로 적용할까요?", parent=self.win):
                return
        self.result = self._value
        self.win.destroy()

    def _cancel(self):
        self.result = None
        self.win.destroy()


def measure_scale_bar(parent, image_path: str) -> Optional[float]:
    """Show the dialog; return µm/px, or None if the user cancelled."""
    dlg = _Dialog(parent, image_path)
    parent.wait_window(dlg.win)
    return dlg.result
