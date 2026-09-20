"""
Run a command with a wall-clock limit and report everything about it.

Used by CI instead of the `timeout` command, which does not exist on macOS.
Always exits 0: the workflow decides pass/fail from the artifacts the command
was supposed to produce, so that a hang or a crash still leaves a readable log
rather than killing the step before it can print anything.

    python tests/run_timeout.py 120 ./dist/App --cli sample.png --csv out.csv
"""

import subprocess
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: run_timeout.py <seconds> <command> [args...]")
        return 0

    limit = float(sys.argv[1])
    cmd = sys.argv[2:]
    print(f"[run] {' '.join(cmd)}   (limit {limit:.0f}s)", flush=True)

    try:
        p = subprocess.run(cmd, timeout=limit, capture_output=True)
        out, err, code, timed_out = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        out, err, code, timed_out = e.stdout, e.stderr, None, True
    except FileNotFoundError:
        print(f"[run] command not found: {cmd[0]}")
        return 0
    except Exception as e:                       # noqa: BLE001
        print(f"[run] could not start: {type(e).__name__}: {e}")
        return 0

    def show(label, blob):
        if not blob:
            print(f"[{label}] (empty)")
            return
        if isinstance(blob, bytes):
            blob = blob.decode("utf-8", errors="replace")
        print(f"[{label}]")
        print(blob.rstrip())

    show("stdout", out)
    show("stderr", err)

    if timed_out:
        print(f"[run] TIMED OUT after {limit:.0f}s — the process hung "
              f"(on Windows this usually means a modal error dialog).")
    else:
        print(f"[run] exit code {code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
