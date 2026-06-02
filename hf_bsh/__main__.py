"""Entry point: argument parsing, the readline REPL, and non-TTY batch mode."""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from . import __version__
from .api import Client
from .completion import complete
from .shell import Shell, ShellError

USAGE = """\
hf-bsh — interactive bucket shell for the Hugging Face Hub.

Usage:
  hf-bsh [options] [<target>]
  hf bsh [options] [<target>]            (when installed as an `hf` extension)

Target:
  <ns>/<name>                bucket to open (read/write)

External sources for `cp`:
  hf://buckets/<ns>/<name>/<path>        other bucket (server-side xet copy)
  hf://datasets/<repo>/<path>            dataset file (server-side xet copy)
  hf://models/<repo>/<path>              model file   (server-side xet copy)

Options:
  --endpoint <URL>           override Hub endpoint (or set $HF_ENDPOINT)
  --token <TOKEN>            override auth token (or set $HF_TOKEN)
  -h, --help                 print this help
  -V, --version              print version

Authentication:
  Reads $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN, then the saved token file
  (`hf auth login`), resolved by huggingface_hub.
"""


class _Args:
    def __init__(self):
        self.target: Optional[str] = None
        self.endpoint: Optional[str] = None
        self.token: Optional[str] = None


def parse_args(argv: List[str]) -> _Args:
    out = _Args()
    it = iter(argv)
    for a in it:
        if a in ("-h", "--help"):
            print(USAGE, end="")
            sys.exit(0)
        elif a in ("-V", "--version"):
            print(f"hf-bsh {__version__}")
            sys.exit(0)
        elif a == "--endpoint":
            out.endpoint = next(it, None) or _die("--endpoint requires a value")
        elif a == "--token":
            out.token = next(it, None) or _die("--token requires a value")
        elif a.startswith("--endpoint="):
            out.endpoint = a[len("--endpoint="):]
        elif a.startswith("--token="):
            out.token = a[len("--token="):]
        elif a.startswith("-"):
            _die(f"unknown option: {a}")
        else:
            if out.target is not None:
                _die(f"unexpected extra argument: {a}")
            out.target = a
    return out


def _die(msg: str):
    sys.stderr.write(f"hf-bsh: {msg}\n\n{USAGE}")
    sys.exit(2)


def _history_path() -> Optional[str]:
    home = os.path.expanduser("~")
    return os.path.join(home, ".hf-bsh_history") if home != "~" else None


def _run_batch(shell: Shell):
    for line in sys.stdin:
        try:
            if shell.run_line(line.rstrip("\n")):
                break
        except ShellError as e:
            sys.stderr.write(f"hf-bsh: {e}\n")
        except Exception as e:  # noqa: BLE001 — surface, don't crash the loop
            sys.stderr.write(f"hf-bsh: {e}\n")


def _setup_readline(shell: Shell):
    try:
        import readline
    except ImportError:
        return None

    def completer(text, state_idx):
        line = readline.get_line_buffer()
        point = readline.get_endidx()
        try:
            matches = complete(shell.state, line, point)
        except Exception:
            matches = []
        # readline replaces the delimited word; our candidates are full tokens.
        begin = readline.get_begidx()
        word = line[begin:point]
        # Strip the leading part of the candidate that readline already has so
        # the visible completion lines up with the cursor word.
        trimmed = []
        head = word[: len(word) - len(text)] if word.endswith(text) else ""
        for m in matches:
            trimmed.append(m[len(head):] if head and m.startswith(head) else m)
        return trimmed[state_idx] if state_idx < len(trimmed) else None

    readline.set_completer_delims(" \t\n")
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")
    hist = _history_path()
    if hist:
        try:
            readline.read_history_file(hist)
        except OSError:
            pass
    return hist


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    client = Client(endpoint=args.endpoint, token=args.token)
    shell = Shell(client)

    if args.target:
        try:
            shell.run_line(f"open {args.target}")
        except ShellError as e:
            sys.stderr.write(f"hf-bsh: {e}\n")

    if not sys.stdin.isatty():
        _run_batch(shell)
        return 0

    print(
        f"hf-bsh {__version__} — type `help` for commands.\n"
        "  open <ns>/<name>              bucket (read/write)\n"
        "  cp hf://datasets/<id>/<path>  <dst>/   pull external data in"
    )
    hist = _setup_readline(shell)

    while True:
        try:
            line = input(shell.prompt())
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("^C")
            continue
        try:
            if shell.run_line(line):
                break
        except ShellError as e:
            sys.stderr.write(f"hf-bsh: {e}\n")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"hf-bsh: {e}\n")

    if hist:
        try:
            import readline

            readline.write_history_file(hist)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
