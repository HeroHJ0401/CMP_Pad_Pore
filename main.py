"""
Entry point for both the GUI and the CLI.

    python main.py                 -> GUI
    python main.py img1.png ...    -> GUI with those files loaded
    python main.py --cli img*.png  -> headless CLI
"""

import os
import sys
import multiprocessing


def _ensure_console_io():
    """
    Give the process usable stdout/stderr before anything prints.

    A PyInstaller --windowed build on Windows has no console: sys.stdout and
    sys.stderr are None, so the first print() raises, and the bootloader shows
    a modal error dialog. That dialog is invisible on a CI runner and the
    process hangs until the job times out.

    So: attach to the console of whatever launched us (cmd/PowerShell) when
    there is one, which makes --cli output appear as the user expects, and
    fall back to os.devnull so printing can never raise.
    """
    if sys.platform == "win32" and (sys.stdout is None or sys.stderr is None):
        try:
            import ctypes
            ATTACH_PARENT_PROCESS = -1
            if ctypes.windll.kernel32.AttachConsole(ATTACH_PARENT_PROCESS):
                for name in ("stdout", "stderr"):
                    if getattr(sys, name) is None:
                        try:
                            setattr(sys, name, open("CONOUT$", "w",
                                                    buffering=1, errors="replace"))
                        except OSError:
                            pass
        except Exception:
            pass

    sink = None
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            if sink is None:
                sink = open(os.devnull, "w")
            setattr(sys, name, sink)
    if sys.stdin is None:
        try:
            sys.stdin = open(os.devnull, "r")
        except OSError:
            pass

    # This program prints Korean. A Windows console defaults to a legacy code
    # page (cp949 / cp1252) that cannot encode it, and the resulting
    # UnicodeEncodeError would abort a batch run — or, in a windowed build,
    # surface as a modal error dialog. Never let an output encoding kill the
    # analysis: the numbers are already in the CSV.
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def main():
    multiprocessing.freeze_support()   # required for PyInstaller onefile builds
    _ensure_console_io()

    argv = sys.argv[1:]
    if argv and argv[0] == "--cli":
        from pore_analyzer.cli import main as cli_main
        sys.exit(cli_main(argv[1:]))

    from pore_analyzer.gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
