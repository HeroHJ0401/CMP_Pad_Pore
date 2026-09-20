"""
Tkinter GUI for CMP pad SEM open-pore fraction analysis.

Drag images onto the window (if tkinterdnd2 is available), drop them on the
.exe icon, or use the 이미지 추가 button. Results appear in a table and the
segmentation overlay is drawn on the right.
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

from .core import Params, analyze_path, load_gray, analyze_array, robustness_sweep, detect_info_bar

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:
    HAS_DND = False

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

COLS = [
    ("file", "파일", 170),
    ("n_objects", "객체수", 60),
    ("open_pore_fraction_pct", "개공률 %", 80),
    ("dark_area_fraction_pct", "암부면적 %", 85),
    ("circularity_median", "원형도 중앙값", 95),
    ("solidity_median", "Solidity 중앙값", 105),
    ("eqdiam_median_um", "등가직경 µm", 90),
    ("object_density_per_mm2", "밀도 /mm²", 80),
    ("concave_ratio_pct", "오목비율 %", 80),
    ("n_rejected_border", "프레임접촉", 75),
    ("border_area_fraction_pct", "접촉면적 %", 85),
    ("otsu_threshold_ref", "Otsu(참고)", 80),
]


class App:
    def __init__(self, root, initial_files=None):
        self.root = root
        root.title("CMP Pad 개공률 분석기")
        root.geometry("1280x800")
        root.minsize(1060, 660)

        self.files: list[str] = []
        self.results: dict[str, object] = {}
        self._preview_img = None
        self._q: queue.Queue = queue.Queue()
        self._busy = False

        self._build_style()
        self._build_widgets()

        if HAS_DND:
            root.drop_target_register(DND_FILES)
            root.dnd_bind("<<Drop>>", self._on_drop)

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

    def _build_widgets(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        # ---- top: files + params --------------------------------------
        top = ttk.Frame(outer)
        top.pack(fill="x")

        # file list
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
        hint = "창에 파일을 끌어다 놓으셔도 됩니다" if HAS_DND else "이미지 추가 버튼을 사용하십시오"
        ttk.Label(fbtn, text=hint, style="Hint.TLabel", wraplength=130).pack(fill="x", pady=(6, 0))

        # parameters
        pbox = ttk.LabelFrame(top, text="분석 조건", padding=6)
        pbox.pack(side="left", fill="y", padx=(8, 0))

        self.v = {
            "pixel_size_um": tk.StringVar(value="0.404"),
            "sigma_px": tk.StringVar(value="1.2"),
            "threshold": tk.StringVar(value="0.45"),
            "min_diam_um": tk.StringVar(value="2.0"),
            "solidity_cut": tk.StringVar(value="0.90"),
            "opening_radius_px": tk.StringVar(value="2"),
            "crop_bottom_px": tk.StringVar(value="0"),
        }
        rows = [
            ("픽셀 크기 (µm/px)", "pixel_size_um"),
            ("Gaussian σ (px)", "sigma_px"),
            ("이진화 임계 (0–1)", "threshold"),
            ("최소 등가직경 (µm)", "min_diam_um"),
            ("Solidity 기준", "solidity_cut"),
            ("Opening 반경 (px)", "opening_radius_px"),
            ("하단 크롭 (px)", "crop_bottom_px"),
        ]
        for i, (label, key) in enumerate(rows):
            r, c = i % 4, i // 4
            ttk.Label(pbox, text=label).grid(row=r, column=c * 2, sticky="w", padx=(0, 6), pady=2)
            ttk.Entry(pbox, textvariable=self.v[key], width=8).grid(row=r, column=c * 2 + 1, pady=2)

        self.var_border = tk.BooleanVar(value=True)
        self.var_autocrop = tk.BooleanVar(value=True)
        ttk.Checkbutton(pbox, text="프레임 접촉 객체 제외",
                        variable=self.var_border).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(pbox, text="SEM 정보바 자동 크롭",
                        variable=self.var_autocrop).grid(row=4, column=2, columnspan=2, sticky="w", pady=(6, 0))

        abox = ttk.Frame(pbox)
        abox.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.btn_run = ttk.Button(abox, text="분석 실행", command=self._run)
        self.btn_run.pack(side="left")
        ttk.Button(abox, text="CSV 저장", command=self._save_csv).pack(side="left", padx=4)
        ttk.Button(abox, text="오버레이 저장", command=self._save_overlay).pack(side="left")
        ttk.Button(abox, text="강건성 스윕", command=self._run_sweep).pack(side="left", padx=4)

        # ---- middle: table + preview ----------------------------------
        mid = ttk.Frame(outer)
        mid.pack(fill="both", expand=True, pady=(8, 0))

        tbox = ttk.LabelFrame(mid, text="결과", padding=4)
        tbox.pack(side="left", fill="both", expand=True)
        self.tree = ttk.Treeview(tbox, columns=[c[0] for c in COLS],
                                 show="headings", height=10)
        for key, title, wdt in COLS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=wdt, anchor="center")
        self.tree.column("file", anchor="w")
        self.tree.pack(side="top", fill="both", expand=True)
        hsb = ttk.Scrollbar(tbox, orient="horizontal", command=self.tree.xview)
        hsb.pack(side="top", fill="x")
        self.tree.config(xscrollcommand=hsb.set)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._show_preview(from_tree=True))

        vbox = ttk.LabelFrame(mid, text="분할 결과 (녹색 = 볼록 개구부, 적색 = 오목/제외)", padding=4)
        vbox.pack(side="left", fill="both", padx=(8, 0))
        self.canvas = tk.Canvas(vbox, width=430, height=330, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.lbl_big = ttk.Label(vbox, text="개공률 —", style="Big.TLabel")
        self.lbl_big.pack(anchor="w", pady=(6, 0))
        self.lbl_sub = ttk.Label(vbox, text="", style="Hint.TLabel", wraplength=420, justify="left")
        self.lbl_sub.pack(anchor="w")

        # ---- status ----------------------------------------------------
        self.status = ttk.Label(outer, text="대기 중", style="Hint.TLabel", anchor="w")
        self.status.pack(fill="x", pady=(6, 0))

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
                if autocrop and p.crop_bottom_px == 0:
                    g0 = load_gray(path, 0)
                    n = detect_info_bar(g0)
                    if n:
                        pp = p.copy_with(crop_bottom_px=n)
                res = analyze_path(path, pp)
                self._q.put(("result", (path, res)))
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
                    path, res = payload
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
        self.status.config(text="강건성 스윕 계산 중…")
        self.root.update_idletasks()
        gray = load_gray(sel, p.crop_bottom_px)
        rows = robustness_sweep(gray, p)
        self.status.config(text="스윕 완료")

        win = tk.Toplevel(self.root)
        win.title(f"강건성 스윕 — {os.path.basename(sel)}")
        win.geometry("520x460")
        ttk.Label(win, padding=8, justify="left", wraplength=490,
                  text=("임계 0.35–0.65 × solidity 0.85–0.95 조합별 개공률입니다. "
                        "절대값은 크게 변하므로 두 조건을 동일 설정에서 비교한 "
                        "대소 관계만 사용하십시오.")).pack(fill="x")
        tv = ttk.Treeview(win, columns=("t", "s", "f", "n"), show="headings")
        for k, t, w in (("t", "임계", 90), ("s", "Solidity", 90),
                        ("f", "개공률 %", 110), ("n", "객체수", 90)):
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
    def _show_preview(self, from_tree=False):
        path = self._selected_path()
        if path is None or path not in self.results:
            return
        res = self.results[path]
        ov = res.overlay
        if ov is None:
            return
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 160)
        im = Image.fromarray(ov)
        scale = min(cw / im.width, ch / im.height)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.LANCZOS)
        self._preview_img = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._preview_img, anchor="center")

        self.lbl_big.config(text=f"개공률 {100*res.open_pore_fraction:.2f} %")
        self.lbl_sub.config(text=(
            f"객체 {res.n_objects}개 · 원형도 중앙값 {res.circularity_median:.3f} · "
            f"Solidity 중앙값 {res.solidity_median:.3f}\n"
            f"시야 {res.field_w_um:.0f} × {res.field_h_um:.0f} µm · "
            f"단순 암부면적률 {100*res.dark_area_fraction:.1f} % (그림자 포함, 참고용)\n"
            f"프레임 접촉 제외 {res.n_rejected_border}개 "
            f"(면적 {100*res.border_area_fraction:.1f} % — 개공률은 과소평가)"))

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
