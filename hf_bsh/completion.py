"""Tab-completion candidate generation. Kept separate from the readline glue
in `__main__` so it can be unit-tested without a terminal.

Each function returns a list of *replacement* strings for the current token
(the word after the last whitespace), mirroring the Rust completer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .shell import COMMANDS, State
from .paths import expand_tilde


def complete(state: State, line: str, point: int) -> List[str]:
    """Top-level dispatch: given the full input `line` and cursor `point`,
    return replacement candidates for the token under the cursor."""
    before = line[:point]
    # Token starts after the last whitespace char (slashes stay inside the token).
    ws = max(before.rfind(c) for c in " \t\n")
    start = ws + 1  # ws == -1 when no whitespace -> start == 0
    token = before[start:]
    prefix = before[:start].strip()

    if not prefix:
        return [c + " " for c in COMMANDS if c.startswith(token)]

    cmd = prefix.split()[0]
    if cmd == "open":
        return _complete_open(state, token)
    if cmd == "cd":
        return [p for p in _complete_remote_path(state, token) if p.endswith("/")]
    if cmd in ("ls", "cat", "du", "find", "tree", "rm"):
        return _complete_remote_path(state, token)
    if cmd in ("mv", "cp"):
        if token.startswith("hf://") or "hf://".startswith(token):
            return _complete_hf_url(state, token)
        return _complete_remote_path(state, token)
    if cmd == "get":
        arg_index = len(prefix.split()) - 1
        return _complete_remote_path(state, token) if arg_index == 0 else _complete_local_path(token)
    if cmd == "put":
        return _complete_local_path(token)
    return []


def _complete_local_path(text: str) -> List[str]:
    if text == "~":
        return ["~/"]
    if "/" in text:
        dir_part, prefix = text.rsplit("/", 1)
        dir_part = dir_part or "/"
    else:
        dir_part, prefix = ".", text
    fs_dir = expand_tilde(dir_part)
    out: List[str] = []
    try:
        entries = list(os.scandir(fs_dir))
    except OSError:
        return out
    head = text[: text.rfind("/") + 1] if "/" in text else ""
    for entry in entries:
        name = entry.name
        if not name.startswith(prefix):
            continue
        suffix = "/" if entry.is_dir() else ""
        out.append(f"{head}{name}{suffix}")
    return out


def _complete_remote_path(state: State, text: str) -> List[str]:
    if "/" in text:
        dir_part, prefix = text.rsplit("/", 1)
    else:
        dir_part, prefix = "", text
    rel = state.remote_prefix(dir_part)
    names = state.listdir(rel)
    head = f"{dir_part}/" if dir_part else ""
    return [f"{head}{n}" for n in names if n.startswith(prefix)]


def _complete_hf_url(state: State, text: str) -> List[str]:
    client = state.client
    if text == "hf://" or "hf://".startswith(text):
        if len(text) < len("hf://"):
            return ["hf://"]
        return ["hf://buckets/", "hf://datasets/", "hf://models/"]
    for seg, lister in (
        ("hf://datasets/", lambda q: client.list_datasets(q or None, 30)),
        ("hf://models/", lambda q: client.list_models(q or None, 30)),
        ("hf://buckets/", lambda q: client.list_buckets(None)),
    ):
        if text.startswith(seg):
            q = text[len(seg):]
            try:
                ids = lister(q)
            except Exception:
                return []
            return [f"{seg}{i}/" for i in ids if i.startswith(q)]
    return []


def _complete_open(state: State, text: str) -> List[str]:
    if text.startswith("buckets/"):
        prefix, q = "buckets/", text[len("buckets/"):]
    else:
        prefix, q = "", text
    try:
        buckets = state.client.list_buckets(None)
    except Exception:
        return []
    return [f"{prefix}{b}" for b in buckets if b.startswith(q)]
