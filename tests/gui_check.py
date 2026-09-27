"""
Drive the real GUI under Xvfb and assert what the user would see.

    xvfb-run -a python3 tests/gui_check.py

Static inspection has missed several bugs in this project (a Tk variable read
from a worker thread, a tooltip that erased itself, a wraplength feedback
loop), so anything touching widgets gets exercised here with root.update()
rather than reasoned about.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

# A headless runner without tkinter or without an X display cannot run this at
# all; that is not a test failure, so say so and exit clean. An assertion that
# actually fails still returns non-zero below.
try:
    import tkinter as tk
    from tkinter import ttk
    _probe = tk.Tk()
    _probe.destroy()
except Exception as _exc:                                  # noqa: BLE001
    print(f"GUI CHECK SKIPPED — 화면을 열 수 없습니다 ({type(_exc).__name__}: {_exc})")
    raise SystemExit(0)

from pore_analyzer import gui as G

FAILED = []
DIALOGS = []


class _NoModal:
    """
    Record dialogs instead of showing them.

    A real messagebox under Xvfb waits for a click that never comes — the same
    way a --windowed build hung a CI runner for 35 minutes. Any test that can
    reach a dialog has to neutralise it first.
    """

    def __getattr__(self, name):
        def fake(title="", message="", **kw):
            DIALOGS.append((name, str(title), str(message)))
            return True
        return fake


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    if not cond:
        FAILED.append(name)


def pump(root, n=6):
    for _ in range(n):
        root.update()
        root.after(6)
        root.update()


# --------------------------------------------------------------- test images
def field(path, pore_frac=0.09, h=420, w=560, seed=1):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 0.72)
    yy, xx = np.ogrid[:h, :w]
    n = max(1, int(pore_frac * h * w / (np.pi * 9 ** 2)))
    for y, x in zip(rng.integers(0, h, n), rng.integers(0, w, n)):
        r = rng.integers(6, 13)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        d = np.sqrt((yy[y0:y1] - y) ** 2 + (xx[:, x0:x1] - x) ** 2)
        prof = 1.0 / (1.0 + np.exp((d - r) / 0.9))
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - prof) + 0.22 * prof
    img = np.clip(img + rng.normal(0, 0.015, img.shape), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path


def blank(path, h=420, w=560):
    """No pores at all — must come back flagged, not reported as a number."""
    rng = np.random.default_rng(9)
    img = np.clip(0.72 + rng.normal(0, 0.015, (h, w)), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path


def wait_idle(app, root, limit=600):
    for _ in range(limit):
        pump(root, 2)
        if not app._busy:
            return True
    return False


def main():
    G.messagebox = _NoModal()
    tmp = tempfile.mkdtemp(prefix="guicheck_")
    good = field(os.path.join(tmp, "good.png"))
    empty = blank(os.path.join(tmp, "empty.png"))

    root = tk.Tk()
    app = G.App(root)
    pump(root)

    # ---------------------------------------------------- threshold locking
    print("\n임계 입력칸 잠금과 '적용 임계' 표시")
    ent = app.param_entries["threshold"]
    check("고정 모드에서는 임계 입력칸이 열려 있다",
          str(ent.cget("state")) == "normal", f"state={ent.cget('state')}")
    check("고정 모드 표시에 입력값이 나온다",
          "0.45" in app.lbl_thr.cget("text"), app.lbl_thr.cget("text"))

    app.var_mode.set("Otsu (일괄)")
    app._on_mode_change()
    pump(root, 5)
    check("Otsu 모드에서는 임계 입력칸이 잠긴다",
          str(ent.cget("state")) == "disabled", f"state={ent.cget('state')}")
    txt = app.lbl_thr.cget("text")
    check("실행 전에는 '분석 실행 후 결정'이라고 알린다", "분석 실행 후 결정" in txt, txt)
    check("입력칸 값이 쓰이지 않는다고 명시한다", "쓰이지 않습니다" in txt, txt)

    app.var_mode.set("고정")
    app._on_mode_change()
    pump(root, 5)
    check("고정으로 되돌리면 입력칸이 다시 열린다",
          str(ent.cget("state")) == "normal")
    app.v["threshold"].set("0.52")
    pump(root, 5)
    check("입력값을 바꾸면 표시도 따라간다",
          "0.52" in app.lbl_thr.cget("text"), app.lbl_thr.cget("text"))
    app.v["threshold"].set("0.45")

    # ------------------------------------------------------------ analysis
    print("\n분석 실행과 품질 경고")
    app._add_files([good, empty])
    pump(root, 5)
    app.var_mode.set("Otsu (이미지별)")
    app._on_mode_change()
    app._run()
    check("분석이 시간 내에 끝났다", wait_idle(app, root))
    check("두 이미지 모두 결과가 나왔다", len(app.results) == 2, str(len(app.results)))

    r_good = app.results.get(good)
    r_empty = app.results.get(empty)
    check("기공이 있는 이미지는 경고가 없다",
          r_good is not None and not r_good.warnings,
          str(getattr(r_good, "warnings", None)))
    check("기공이 없는 이미지는 분리도 경고가 붙는다",
          r_empty is not None and any("분리도" in w for w in r_empty.warnings),
          str(getattr(r_empty, "warnings", None)))
    check("경고 이미지의 행에 flagged 태그가 붙는다",
          "flagged" in app.tree.item(empty, "tags"),
          str(app.tree.item(empty, "tags")))
    check("정상 이미지의 행에는 태그가 없다",
          "flagged" not in app.tree.item(good, "tags"))
    check("경고 줄이 화면에 떠 있다",
          "주의" in app.lbl_warn.cget("text"), app.lbl_warn.cget("text"))
    txt = app.lbl_thr.cget("text")
    check("이미지별 Otsu는 범위로 표시된다",
          "–" in txt and "이미지마다 다름" in txt, txt)

    # the new columns must actually carry values
    print("\n새 열의 값")
    row = r_good.summary_row()
    for key in ("threshold_source", "separability", "dynamic_range", "gray_levels"):
        check(f"{key} 에 값이 있다", row.get(key) not in ("", None), repr(row.get(key)))

    # ------------------------------------------------------- batch identity
    print("\nOtsu(일괄)은 순서에 관계없이 같은 임계값")
    app.var_mode.set("Otsu (일괄)")
    app._on_mode_change()
    seen = []
    for order in ([good, empty], [empty, good]):
        app.files = list(order)
        app._run()
        wait_idle(app, root)
        seen.append(round(app.results[good].effective_threshold, 6))
    check("순서를 바꿔도 임계값이 동일하다", seen[0] == seen[1], f"{seen}")

    # -------------------------------------------------------------- paste
    print("\n클립보드 붙여넣기")
    import PIL.ImageGrab as IG

    bitmap = Image.open(good).convert("RGB")
    IG.grabclipboard = lambda: bitmap          # a bitmap on the clipboard
    n0 = len(app.files)
    app._paste_from_clipboard()
    pump(root, 5)
    check("비트맵을 붙여넣으면 목록에 1개 늘어난다",
          len(app.files) == n0 + 1, f"{n0} -> {len(app.files)}")
    pasted = app.files[-1]
    check("임시 PNG 파일로 저장된다",
          os.path.isfile(pasted) and pasted.lower().endswith(".png"), pasted)
    check("이름이 '클립보드'로 시작한다",
          os.path.basename(pasted).startswith("클립보드"), os.path.basename(pasted))

    IG.grabclipboard = lambda: [empty]         # files copied in the file manager
    app.files = [good]
    app.flist.delete(*app.flist.get_children())
    app.flist.insert("", "end", iid=good, values=(os.path.basename(good), "", ""))
    app._paste_from_clipboard()
    pump(root, 5)
    check("복사된 파일 경로도 받아들인다", empty in app.files, str(app.files))

    # focus in an entry -> the handler must step aside so text paste still works
    n1 = len(app.files)
    IG.grabclipboard = lambda: bitmap
    ent.focus_set()
    pump(root, 5)
    fake = type("E", (), {})()
    app._paste_from_clipboard(fake)
    pump(root, 5)
    check("입력칸에 포커스가 있으면 이미지를 붙여넣지 않는다",
          len(app.files) == n1, f"{n1} -> {len(app.files)}")

    IG.grabclipboard = lambda: None
    app.root.clipboard_clear()
    app.root.clipboard_append("그냥 글자")
    app.flist.focus_set()
    n2 = len(app.files)
    app._paste_from_clipboard()
    pump(root, 5)
    check("클립보드에 이미지가 없으면 목록이 그대로다", len(app.files) == n2)
    check("이미지가 없다는 안내를 띄운다",
          any("클립보드" in t for _k, t, _m in DIALOGS), str(DIALOGS[-1:]))

    # ------------------------------------------------------------ tooltips
    print("\n열 머리글 툴팁")
    shown = {}
    app.tip._show = lambda text, x, y: shown.__setitem__("text", text)

    def wait_tip(limit=80):
        for _ in range(limit):          # the tooltip is deliberately delayed
            pump(root, 4)
            if shown.get("text"):
                return True
        return False

    # the popup machinery itself, through the real delay
    app.tip.hide()
    shown.clear()
    app.tip.request("col:file", "파일\n\n설명", 10, 10)
    check("툴팁이 지연 후 실제로 뜬다", wait_tip(), repr(shown.get("text"))[:40])

    # the heading -> COL_HELP lookup, for the new columns. The real columns sit
    # off to the right of the visible area, so identify_* is stubbed rather than
    # scrolling the table; what is under test is the mapping, not Tk geometry.
    idx = [c[0] for c in G.COLUMNS].index("separability")
    app.tree.identify_region = lambda x, y: "heading"
    app.tree.identify_column = lambda x: f"#{idx + 1}"
    for key in ("separability", "dynamic_range", "gray_levels", "threshold_source"):
        idx = [c[0] for c in G.COLUMNS].index(key)
        app.tip.hide()
        shown.clear()
        ev = type("E", (), {"x": 5, "y": 5, "x_root": 100, "y_root": 100})()
        app._on_tree_motion(ev)
        ok = wait_tip()
        title = dict((k, t) for k, t, _w in G.COLUMNS)[key]
        check(f"'{title}' 머리글 설명이 뜬다",
              ok and title in shown.get("text", ""),
              repr(shown.get("text", ""))[:45])

    root.destroy()
    print("\n" + ("GUI CHECK PASSED" if not FAILED
                  else f"GUI CHECK FAILED: {len(FAILED)}건 — " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
