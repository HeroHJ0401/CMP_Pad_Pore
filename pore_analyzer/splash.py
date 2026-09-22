"""
A loading window shown while the heavy libraries import.

numpy, scipy, scikit-image and Pillow together take several seconds to import,
and in a frozen build that happens on every launch. Without a splash the user
gets a few seconds of nothing after double-clicking and reasonably concludes
the program did not start.

This module deliberately imports nothing but tkinter, so it can be shown
BEFORE the expensive imports begin.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

__all__ = ["Splash", "start_gui_with_splash"]

BG = "#1f2430"
FG = "#e8ecf3"
DIM = "#98a2b3"
ACCENT = "#4a9eff"


class Splash:
    """Small centred window with a title and an animated bar."""

    def __init__(self, root, app_name="", message="불러오는 중…"):
        self.root = root
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        try:
            self.win.attributes("-topmost", True)
        except tk.TclError:
            pass

        frame = tk.Frame(self.win, bg=BG, padx=34, pady=26,
                         highlightthickness=1, highlightbackground="#3a4254")
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=app_name, bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack()
        self.lbl = tk.Label(frame, text=message, bg=BG, fg=DIM,
                            font=("Segoe UI", 10))
        self.lbl.pack(pady=(8, 14))

        self.canvas = tk.Canvas(frame, width=260, height=5, bg="#2b3242",
                                highlightthickness=0)
        self.canvas.pack()
        self._x = -70
        self._job = None
        self._animate()

        self.win.update_idletasks()
        w, h = self.win.winfo_width(), self.win.winfo_height()
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        self.win.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2 - 40}")
        self.win.update()

    def _animate(self):
        try:
            self.canvas.delete("all")
            self.canvas.create_rectangle(self._x, 0, self._x + 70, 5,
                                         fill=ACCENT, outline="")
            self._x += 7
            if self._x > 260:
                self._x = -70
            self._job = self.win.after(28, self._animate)
        except tk.TclError:
            self._job = None            # window went away mid-animation

    def message(self, text):
        try:
            self.lbl.config(text=text)
            self.win.update()
        except tk.TclError:
            pass

    def close(self):
        if self._job is not None:
            try:
                self.win.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        try:
            self.win.destroy()
        except tk.TclError:
            pass


def _boot_splash(action, text=None):
    """
    Drive PyInstaller's own splash, which covers the gap this module cannot:
    the seconds a onefile build spends unpacking itself before Python starts.

    The `pyi_splash` module exists only inside a frozen build that was given
    --splash, and PyInstaller does not support it on macOS, so every call is
    optional by design.
    """
    try:
        import pyi_splash
    except Exception:
        return
    try:
        if action == "text" and text:
            pyi_splash.update_text(text)
        elif action == "close":
            pyi_splash.close()
    except Exception:
        pass


def start_gui_with_splash(app_name_fallback="", initial_files=None):
    """
    Show the splash, import the GUI behind it, then hand over to the app.

    The import runs on a worker thread so the splash keeps animating; the
    main thread only ever touches Tk. If anything about that goes wrong we
    fall back to importing inline, which is slower to look at but correct.
    """
    import queue
    import threading

    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    root.withdraw()

    _boot_splash("text", "라이브러리를 불러오는 중…")
    splash = Splash(root, app_name_fallback or "CMP Pad 개공율 분석기")
    # Our own window is up now, so hand over from the bootloader's splash.
    _boot_splash("close")

    box: "queue.Queue" = queue.Queue()

    def load():
        try:
            from .gui import App, APP_NAME
            box.put(("ok", (App, APP_NAME)))
        except BaseException as exc:                      # noqa: BLE001
            box.put(("err", exc))

    threading.Thread(target=load, daemon=True).start()

    payload = None
    while payload is None:
        try:
            payload = box.get_nowait()
        except queue.Empty:
            root.update()
            root.after(20)
            root.update()

    kind, value = payload
    if kind == "err":
        _boot_splash("close")
        splash.close()
        root.deiconify()
        from tkinter import messagebox
        messagebox.showerror("시작 실패", f"{type(value).__name__}: {value}")
        raise value

    App, APP_NAME = value
    splash.message("화면을 준비하는 중…")
    root.title(APP_NAME)
    App(root, initial_files=initial_files or [])
    splash.close()
    root.deiconify()
    try:
        root.lift()
        root.focus_force()
    except tk.TclError:
        pass
    root.mainloop()
