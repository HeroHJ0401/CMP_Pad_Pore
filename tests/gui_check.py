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

    # ------------------------------------------------- paste key recognition
    # The reported bug: the button worked, Ctrl+V did nothing. With a Hangul
    # layout active Tk reports the keysym of the character the layout produces,
    # so a <Control-v> binding never matches. Match on the hardware key code.
    print("\n붙여넣기 키 인식")
    def ev(**kw):
        e = type("E", (), {})()
        e.keysym = kw.get("keysym", "")
        e.keycode = kw.get("keycode", 0)
        return e

    check("영문 자판의 v를 받는다", app._looks_like_paste(ev(keysym="v", keycode=86)))
    check("한글 자판이라 keysym이 달라도 키코드로 받는다",
          app._looks_like_paste(ev(keysym="ㅍ", keycode=86)))
    check("맥 키코드도 받는다", app._looks_like_paste(ev(keysym="ㅍ", keycode=9)))
    check("<<Paste>> 가상 이벤트를 받는다", app._looks_like_paste(ev(keysym="??")))
    check("Ctrl+C 같은 다른 조합은 무시한다",
          not app._looks_like_paste(ev(keysym="c", keycode=67)))
    check("Ctrl+A 도 무시한다", not app._looks_like_paste(ev(keysym="a", keycode=65)))

    IG.grabclipboard = lambda: bitmap
    app.flist.focus_set()
    pump(root, 3)
    app._last_paste_ms = 0.0
    n3 = len(app.files)
    app._on_paste_event(ev(keysym="ㅍ", keycode=86))
    pump(root, 3)
    check("한글 자판 키 이벤트로 실제로 붙여넣어진다",
          len(app.files) == n3 + 1, f"{n3} -> {len(app.files)}")
    # the overlapping bindings all fire for one keystroke; only one may act
    n4 = len(app.files)
    app._on_paste_event(ev(keysym="v", keycode=86))
    app._on_paste_event(ev(keysym="??"))
    pump(root, 3)
    check("겹치는 바인딩이 여러 번 붙여넣지 않는다",
          len(app.files) == n4, f"{n4} -> {len(app.files)}")

    # ---------------------------------------------------------- split option
    print("\n객체화 방식")
    app.var_objectify.set("거리 분할")
    app.v["split_depth_um"].set("2.5")
    p = app._params()
    check("거리 분할이 분석 조건으로 전달된다",
          p.objectify == "distance" and abs(p.split_depth_um - 2.5) < 1e-9,
          f"{p.objectify}, {p.split_depth_um}")
    app.var_objectify.set("지형 분할")
    app.v["terrain_depth"].set("0.06")
    p = app._params()
    check("지형 분할도 전달된다",
          p.objectify == "terrain" and abs(p.terrain_depth - 0.06) < 1e-9,
          f"{p.objectify}, {p.terrain_depth}")
    app.var_objectify.set("연결 성분")
    check("되돌리면 연결 성분", app._params().objectify == "cc")

    # the depth is a judgement made by eye, so the picker has to actually run
    # the image at several depths and write the chosen one back
    print("\n지형 깊이 비교 창")
    app.tree.selection_set(good)
    pump(root, 3)
    app.v["terrain_depth"].set("0.08")
    before = len(root.winfo_children())
    app._compare_terrain()
    pump(root, 6)
    wins = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    check("비교 창이 뜬다", len(wins) >= 1, f"{before} -> {len(root.winfo_children())}")
    if wins:
        win = wins[-1]
        btns = []

        def walk(w):
            for ch in w.winfo_children():
                if isinstance(ch, ttk.Button) and ch.cget("text") in (
                        "전체에 적용", "이 이미지만"):
                    btns.append((str(ch.cget("text")), ch))
                walk(ch)
        walk(win)
        check("깊이마다 적용 버튼 두 개가 있다", len(btns) >= 10, str(len(btns)))
        check("썸네일이 살아 있다", len(app._terrain_thumbs) >= 5,
              str(len(app._terrain_thumbs)))
        one = [b for n, b in btns if n == "이 이미지만"]
        allb = [b for n, b in btns if n == "전체에 적용"]
        if allb:
            allb[0].invoke()
            pump(root, 4)
            check("'전체에 적용'이 기본 깊이를 바꾼다",
                  app._params().objectify == "terrain"
                  and abs(app._params().terrain_depth - 0.02) < 1e-9
                  and not app.depth_info,
                  f"{app.v['terrain_depth'].get()}, per-image={app.depth_info}")
        # reopen and take the per-image branch
        app._compare_terrain()
        pump(root, 6)
        wins2 = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
        btns2 = []
        def walk2(w):
            for ch in w.winfo_children():
                if isinstance(ch, ttk.Button) and ch.cget("text") == "이 이미지만":
                    btns2.append(ch)
                walk2(ch)
        if wins2:
            walk2(wins2[-1])
        if btns2:
            btns2[-1].invoke()
            pump(root, 4)
            check("'이 이미지만'은 그 파일에만 붙는다",
                  good in app.depth_info
                  and abs(app._depth_for(good) - app.depth_info[good]) < 1e-9
                  and abs(app._default_depth() - 0.02) < 1e-9,
                  f"{app.depth_info}")
            check("다른 이미지는 기본값을 그대로 쓴다",
                  abs(app._depth_for(empty) - app._default_depth()) < 1e-9)
            app._check_mixed_scales()
            pump(root, 3)
            check("깊이가 섞이면 경고가 뜬다",
                  "지형 깊이 혼재" in app.lbl_mixed.cget("text"),
                  app.lbl_mixed.cget("text")[:60])
            app.depth_info.clear()
        for w in [x for x in root.winfo_children() if isinstance(x, tk.Toplevel)]:
            w.destroy()
        pump(root, 3)
    app.var_objectify.set("연결 성분")

    # ----------------------------------------- objectify: greying and example
    print("\n객체화 부속 동작")
    check("임계 모드는 셋뿐이다 (3계급 제거)",
          len(G.THRESHOLD_MODES) == 3 and
          not any("3계급" in n for n, _k in G.THRESHOLD_MODES),
          str([n for n, _k in G.THRESHOLD_MODES]))

    ent_t = app.param_entries["terrain_depth"]
    ent_d = app.param_entries["split_depth_um"]
    app.var_objectify.set("연결 성분"); app._on_objectify_change(); pump(root, 3)
    check("연결 성분이면 두 깊이 칸 모두 잠긴다",
          str(ent_t.cget("state")) == "disabled"
          and str(ent_d.cget("state")) == "disabled")
    app.var_objectify.set("지형 분할"); app._on_objectify_change(); pump(root, 3)
    check("지형 분할이면 지형 깊이만 열린다",
          str(ent_t.cget("state")) == "normal"
          and str(ent_d.cget("state")) == "disabled")
    app.var_objectify.set("거리 분할"); app._on_objectify_change(); pump(root, 3)
    check("거리 분할이면 분리 깊이만 열린다",
          str(ent_d.cget("state")) == "normal"
          and str(ent_t.cget("state")) == "disabled")
    app.var_objectify.set("지형 분할"); app._on_objectify_change(); pump(root, 3)

    # the list column must follow the global value, or it reads as "not applied"
    app.v["terrain_depth"].set("0.11")
    pump(root, 4)
    cell_txt = app.flist.item(good, "values")[2]
    check("기본값을 바꾸면 목록의 지형깊이도 따라간다",
          "0.11" in str(cell_txt), str(cell_txt))

    # the per-image dialog, the same shape as the pixel-size one
    app.flist.selection_set(good)
    pump(root, 3)
    dlg = G._TerrainDepthDialog(app, [good])
    dlg.var.set("0.15")
    dlg._apply()
    pump(root, 4)
    check("대화상자가 이 이미지에만 값을 넣는다",
          abs(app._depth_for(good) - 0.15) < 1e-9
          and abs(app._depth_for(empty) - 0.11) < 1e-9,
          f"{app.depth_info}")
    check("목록에 괄호 없이 표시된다",
          str(app.flist.item(good, "values")[2]).strip() == "0.15",
          str(app.flist.item(good, "values")[2]))
    dlg2 = G._TerrainDepthDialog(app, [good])
    dlg2._clear()
    pump(root, 4)
    check("'기본값 따르기'가 되돌린다",
          good not in app.depth_info
          and abs(app._depth_for(good) - 0.11) < 1e-9)
    app.v["terrain_depth"].set("0.08")
    app.var_objectify.set("연결 성분"); app._on_objectify_change(); pump(root, 3)

    # the worked example that the 객체화 tooltip shows
    panels = app._objectify_panels()
    check("객체화 툴팁 예시가 세 장 만들어진다", len(panels) == 3, str(len(panels)))
    check("예시마다 방식 이름과 수치가 붙는다",
          all(any(k in cap for k in ("연결 성분", "거리 분할", "지형 분할"))
              and "Pore" in cap for _img, cap in panels),
          str([c.splitlines()[0] for _i, c in panels]))
    caps = [c.splitlines()[0] for _i, c in panels]
    check("세 방식이 모두 나온다",
          caps == ["연결 성분", "거리 분할", "지형 분할"], str(caps))

    # ---------------------------------------------------- appearance + hover
    print("\n모양과 요약 호버")
    import tkinter.font as tkfont
    fams = {f.lower() for f in tkfont.families(root)}
    check("해석된 폰트가 실제로 설치되어 있다",
          app.font_family == "TkDefaultFont" or app.font_family.lower() in fams,
          app.font_family)
    check("창 배경이 거의 흰색이다",
          str(root.cget("background")) == G.UI_BG, str(root.cget("background")))
    st = ttk.Style()
    big = str(st.lookup("Big.TLabel", "font"))
    check("제목은 본문보다 크고 굵다 (option_add가 덮어쓰지 않는다)",
          "16" in big and "bold" in big and app.lbl_big.winfo_reqheight() > 24,
          f"{big} / h={app.lbl_big.winfo_reqheight()}")

    # the grey block is gone; its content has to be reachable on hover
    check("회색 요약 블록이 사라졌다", not hasattr(app, "lbl_sub"))
    res0 = app.results.get(good)
    detail = app._detail_text(res0)
    for must in ("Pore 개공율", "임계", "밝기 분리도", "시야", "탈락"):
        check(f"호버 설명에 '{must}'가 있다", must in detail)
    seen_big = {}
    real_show = app.tip._show
    app.tip._show = lambda text, x, y, panels_fn=None: seen_big.__setitem__("t", text)
    app.tip.hide()
    app.tree.selection_set(good)
    pump(root, 3)
    ev2 = type("E", (), {"x_root": 100, "y_root": 100})()
    app._show_big_tip(ev2)
    got = False
    for _ in range(80):
        pump(root, 4)
        if seen_big.get("t"):
            got = True
            break
    check("제목에 마우스를 올리면 설명이 뜬다",
          got and "Pore 개공율" in seen_big.get("t", ""),
          repr(seen_big.get("t", ""))[:40])
    app.tip._show = real_show
    app.tip.hide()

    # ------------------------------------------------------------ tooltips
    print("\n열 머리글 툴팁")
    shown = {}
    app.tip._show = lambda text, x, y, panels_fn=None: (
        shown.__setitem__("text", text),
        shown.__setitem__("panels", panels_fn))

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
