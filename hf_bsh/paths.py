"""Pure path / URL helpers — no network, no I/O. Ported from the Rust `shell.rs`
path utilities so the behaviour (and the tests) match 1:1."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

# hf:// repo kinds, matching the wire `sourceRepoType` vocabulary.
KIND_BUCKET = "bucket"
KIND_DATASET = "dataset"
KIND_MODEL = "model"

# Plural path segment used in `hf://<segment>/...` URLs.
_KIND_TO_SEGMENT = {
    KIND_BUCKET: "buckets",
    KIND_DATASET: "datasets",
    KIND_MODEL: "models",
}


def kind_segment(kind: str) -> str:
    """`"bucket"` -> `"buckets"` etc. (the plural used in hf:// URLs)."""
    return _KIND_TO_SEGMENT[kind]


def has_glob_chars(s: str) -> bool:
    return any(c in s for c in "*?[")


def basename(p: str) -> str:
    """Final path component (no trailing-slash handling — strip first if needed)."""
    return p.rsplit("/", 1)[-1] if "/" in p else p


def normalize_path(p: str) -> str:
    """Collapse `.`/`..` segments in a slash-joined path, drop empties.

    Leading `/` is not preserved (callers pass repo-relative paths). `..` that
    would escape root clamps to root (empty string)."""
    out: list[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if out:
                out.pop()
        else:
            out.append(seg)
    return "/".join(out)


def split_cmd(line: str) -> Tuple[str, str]:
    """Split a REPL line into `(command, rest)` on the first run of whitespace."""
    stripped = line.lstrip()
    for i, c in enumerate(stripped):
        if c.isspace():
            return stripped[:i], stripped[i:].lstrip()
    return stripped, ""


def expand_tilde(s: str) -> str:
    """Expand a leading `~` / `~/` to the user's home dir. `~user` is not
    supported (a REPL is unlikely to need it); everything else is returned as-is."""
    if s == "~":
        home = os.path.expanduser("~")
        return home if home != "~" else s
    if s.startswith("~/"):
        home = os.path.expanduser("~")
        if home != "~":
            return home.rstrip("/") + "/" + s[2:]
    return s


@dataclass
class HfUrl:
    kind: str  # KIND_BUCKET | KIND_DATASET | KIND_MODEL
    repo_id: str
    path: str


def parse_hf_url(s: str) -> Optional[HfUrl]:
    """Parse `hf://{buckets,datasets,models}/<id>/<path>`.

    Returns `None` when `s` isn't an `hf://` URL (caller treats it as a
    bucket-relative path), raises `ValueError` when it is but is malformed."""
    if not s.startswith("hf://"):
        return None
    rest = s[len("hf://"):]
    if rest.startswith("buckets/"):
        kind, after = KIND_BUCKET, rest[len("buckets/"):]
    elif rest.startswith("datasets/"):
        kind, after = KIND_DATASET, rest[len("datasets/"):]
    elif rest.startswith("models/"):
        kind, after = KIND_MODEL, rest[len("models/"):]
    else:
        raise ValueError(
            f"hf url: must start with hf://{{buckets,datasets,models}}/ (got {s!r})"
        )

    if kind == KIND_BUCKET:
        # Buckets always need `<ns>/<name>/<path>`: two id segments then a path.
        parts = after.split("/", 2)
        ns = parts[0] if len(parts) > 0 else ""
        name = parts[1] if len(parts) > 1 else ""
        path = parts[2] if len(parts) > 2 else ""
        if not ns or not name:
            raise ValueError(f"hf url: bucket id must be <ns>/<name> (got {s!r})")
        return HfUrl(kind, f"{ns}/{name}", path)

    # Dataset/model ids are `<name>` (legacy, e.g. `squad`) or `<ns>/<name>`.
    # Heuristic mirrors the Rust port:
    #   >=2 slashes -> id is first two segments (modern form),
    #   exactly 1   -> id is the first segment (legacy form).
    # A single-segment id with a nested path (e.g. `squad/a/b`) is misread, but
    # that shape isn't in common use on the hub.
    slashes = after.count("/")
    if slashes == 0:
        raise ValueError(f"hf url: missing <path> after repo id (got {s!r})")
    if slashes == 1:
        repo_id, path = after.split("/", 1)
        if not repo_id or not path:
            raise ValueError(f"hf url: empty repo id or path (got {s!r})")
        return HfUrl(kind, repo_id, path)
    ns, name, path = after.split("/", 2)
    if not ns or not name or not path:
        raise ValueError(f"hf url: empty segment in {s!r}")
    return HfUrl(kind, f"{ns}/{name}", path)
