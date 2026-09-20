"""
Entry point for both the GUI and the CLI.

    python main.py                 -> GUI
    python main.py img1.png ...    -> GUI with those files loaded
    python main.py --cli img*.png  -> headless CLI
"""

import sys
import multiprocessing


def main():
    multiprocessing.freeze_support()   # required for PyInstaller onefile builds

    argv = sys.argv[1:]
    if argv and argv[0] == "--cli":
        from pore_analyzer.cli import main as cli_main
        sys.exit(cli_main(argv[1:]))

    from pore_analyzer.gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
