"""Interactive bucket shell: state, command dispatch, and tab-completion.

Ported from the Rust `shell.rs`. The command surface and path semantics are
kept identical; the networking now goes through `huggingface_hub` via
`hf_bsh.api.Client`."""

from __future__ import annotations

import fnmatch
import glob as globmod
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import fmt
from .api import Client, TreeEntry
from .paths import (
    KIND_BUCKET,
    KIND_DATASET,
    KIND_MODEL,
    basename,
    expand_tilde,
    has_glob_chars,
    kind_segment,
    normalize_path,
    parse_hf_url,
    split_cmd,
)

CAT_MAX_SIZE = 1 << 20  # 1 MiB

COMMANDS = [
    "open", "cd", "pwd", "ls", "cat", "du", "find", "tree", "rm", "mv", "cp",
    "put", "get", "refresh", "help", "exit", "quit",
]


class ShellError(Exception):
    """A user-facing error. Printed as `hf-bsh: <message>` without a traceback."""


def _bail(msg: str):
    raise ShellError(msg)


def _dirs_first(entries):
    """Sort key helper: directories before files, then lexicographic by path."""
    return sorted(entries, key=lambda e: (not e[0], e[1]))


class State:
    """Mutable shell state shared between the REPL loop and the completer.

    An empty `bucket_id` means "nothing opened" — the shell is bucket-only."""

    def __init__(self, client: Client):
        self.client = client
        self.bucket_id = ""
        self.cwd = ""
        self.ls_cache: Dict[str, List[str]] = {}

    # ---- small predicates ----

    def is_opened(self) -> bool:
        return bool(self.bucket_id)

    def ensure_opened(self, op: str):
        if not self.is_opened():
            _bail(f"{op}: nothing opened")

    def url_root(self) -> str:
        return f"hf://buckets/{self.bucket_id}" if self.is_opened() else "hf://"

    def prompt(self) -> str:
        if not self.is_opened():
            return "hf> "
        shown = f"/{self.cwd}" if self.cwd else ""
        return f"hf:{self.bucket_id}{shown}> "

    def invalidate_cache(self):
        self.ls_cache.clear()

    # ---- path resolution ----

    def remote_prefix(self, p: str) -> str:
        if p.startswith("/"):
            raw = p[1:]
        elif p == "":
            raw = self.cwd
        elif self.cwd == "":
            raw = p
        else:
            raw = f"{self.cwd}/{p}"
        return normalize_path(raw)

    def iter_tree(self, rel_dir: str, recursive: bool) -> List[TreeEntry]:
        self.ensure_opened("tree")
        prefix = f"{rel_dir.rstrip('/')}/" if rel_dir else None
        return self.client.list_bucket_tree(self.bucket_id, prefix, recursive)

    def listdir(self, rel_dir: str) -> List[str]:
        """Immediate children of `rel_dir` as raw names (trailing `/` on dirs). Cached."""
        if not self.is_opened():
            return []
        if rel_dir in self.ls_cache:
            return list(self.ls_cache[rel_dir])
        base = f"{rel_dir.rstrip('/')}/" if rel_dir else ""
        names: List[str] = []
        try:
            entries = self.iter_tree(rel_dir, False)
        except Exception:
            return []
        for e in entries:
            tail = e.path[len(base):] if base and e.path.startswith(base) else e.path
            tail = tail.strip("/")
            if not tail:
                continue
            names.append(f"{tail}/" if e.is_dir else tail)
        self.ls_cache[rel_dir] = list(names)
        return names

    def stat(self, rel: str) -> Optional[Tuple[bool, Optional[int]]]:
        """`(is_dir, size)` for `rel`, or `None` if it doesn't exist. Root is a dir."""
        if rel == "":
            return (True, None)
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        for e in self.iter_tree(parent, False):
            if e.path == rel:
                return (e.is_dir, e.size)
        return None

    def expand_recursive(self, rel: str) -> List[str]:
        return [e.path for e in self.iter_tree(rel, True) if not e.is_dir]

    def glob_match(self, arg: str) -> List[TreeEntry]:
        """Match `arg` (with `*`/`?`/`[...]` in the final component only) against
        the entries of its literal parent directory. zsh-style "no match" error."""
        if "/" in arg:
            dir_part, leaf_pat = arg.rsplit("/", 1)
        else:
            dir_part, leaf_pat = "", arg
        if has_glob_chars(dir_part):
            _bail(f"glob: patterns only supported in the final path component (got {arg!r})")
        base_rel = self.remote_prefix(dir_part)
        prefix = f"{base_rel}/" if base_rel else ""
        out = []
        for e in self.iter_tree(base_rel, False):
            leaf = (e.path[len(prefix):] if prefix and e.path.startswith(prefix) else e.path).strip("/")
            if leaf and fnmatch.fnmatchcase(leaf, leaf_pat):
                out.append(e)
        if not out:
            _bail(f"no match: {arg}")
        return out

    def resolve_targets(self, arg: str) -> List[str]:
        if has_glob_chars(arg):
            return [e.path for e in self.glob_match(arg)]
        return [self.remote_prefix(arg)]


def resolve_entries(st: State, arg: str) -> List[TreeEntry]:
    """`arg` -> list of `TreeEntry`, handling glob / empty-arg / single literal
    uniformly. Literal paths are stat'd so callers can tell files from dirs."""
    if has_glob_chars(arg):
        return st.glob_match(arg)
    rel = st.remote_prefix(arg)
    if rel == "":
        return [TreeEntry(path="", is_dir=True, size=None, mtime=None)]
    s = st.stat(rel)
    if s is None:
        _bail(f"not found: {rel}")
    is_dir, size = s
    return [TreeEntry(path=rel, is_dir=is_dir, size=size, mtime=None)]


# Where a cp/mv source physically lives.
class _Origin:
    OWN = "own"
    EXTERNAL = "external"


class Shell:
    def __init__(self, client: Client):
        self.state = State(client)

    def prompt(self) -> str:
        return self.state.prompt()

    def run_line(self, line: str) -> bool:
        """Run one line. Returns True if the shell should exit."""
        line = line.strip()
        if not line:
            return False
        cmd, rest = split_cmd(line)
        handler = {
            "open": self.do_open,
            "cd": self.do_cd,
            "pwd": lambda _r: self.do_pwd(),
            "ls": self.do_ls,
            "cat": self.do_cat,
            "du": self.do_du,
            "find": self.do_find,
            "tree": self.do_tree,
            "rm": self.do_rm,
            "mv": self.do_mv,
            "cp": self.do_cp,
            "put": self.do_put,
            "get": self.do_get,
            "refresh": self._do_refresh,
            "help": lambda _r: print_help(),
            "?": lambda _r: print_help(),
        }.get(cmd)
        if cmd in ("exit", "quit"):
            return True
        if handler is None:
            _bail(f"unknown command: {cmd}")
        handler(rest)
        return False

    def _do_refresh(self, _rest: str):
        self.state.invalidate_cache()
        print("cache cleared")

    # ---------------- commands ----------------

    def do_open(self, arg: str):
        arg = arg.strip()
        if not arg:
            _bail("usage: open <ns>/<name>")
        if arg.startswith("hf://"):
            _bail(
                f"open: target must be a bucket id (got {arg[len('hf://'):]!r}); "
                "`hf://…` URLs are only valid as `cp` sources"
            )
        for bad in ("datasets/", "models/"):
            if arg.startswith(bad):
                rest = arg[len(bad):]
                _bail(
                    f"open: only buckets can be opened (got {arg!r}); to pull from a "
                    f"dataset/model, use `cp hf://{bad}{rest} <dst>` from an opened bucket"
                )
        rest = arg[len("buckets/"):] if arg.startswith("buckets/") else arg
        parts = rest.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            _bail(f"open: buckets require <ns>/<name> (got {arg!r})")
        st = self.state
        st.bucket_id = rest
        st.cwd = ""
        st.invalidate_cache()

    def do_cd(self, arg: str):
        arg = arg.strip().rstrip("/")
        st = self.state
        st.cwd = "" if not arg else st.remote_prefix(arg)
        st.invalidate_cache()

    def do_pwd(self):
        st = self.state
        print(f"{st.url_root()}/{st.cwd}".rstrip("/"))

    def do_ls(self, arg: str):
        st = self.state
        if not st.is_opened():
            print("nothing opened")
            return
        arg = arg.strip()
        if has_glob_chars(arg):
            matches = [(e.is_dir, e.path, e.size, e.mtime) for e in st.glob_match(arg)]
            for is_dir, path, size, mtime in _dirs_first(matches):
                leaf = basename(path.rstrip("/"))
                print(fmt.fmt_entry(is_dir, size, mtime, leaf))
            return
        rel = st.remote_prefix(arg)
        base = f"{rel.rstrip('/')}/" if rel else ""
        items = []
        for e in st.iter_tree(rel, False):
            tail = (e.path[len(base):] if base and e.path.startswith(base) else e.path).strip("/")
            if not tail:
                continue
            items.append((e.is_dir, tail, e.size, e.mtime))
        for is_dir, name, size, mtime in _dirs_first(items):
            print(fmt.fmt_entry(is_dir, size, mtime, name))

    def do_cat(self, arg: str):
        st = self.state
        if not st.is_opened():
            print("nothing opened")
            return
        arg = arg.strip()
        if not arg:
            _bail("cat: missing path")
        if has_glob_chars(arg):
            matches = st.glob_match(arg)
            if len(matches) != 1:
                _bail(
                    f"cat: pattern {arg!r} matches {len(matches)} entries; "
                    "only single-file cat supported"
                )
            rel = matches[0].path
        else:
            rel = st.remote_prefix(arg)
        display = rel if rel else "/"
        s = st.stat(rel)
        if s is None:
            _bail(f"cat: {display}: not found")
        is_dir, size = s
        if is_dir:
            _bail(f"cat: {display}: is a directory")
        if size is not None and size > CAT_MAX_SIZE:
            _bail(
                f"cat: {rel}: {fmt.fmt_size(size)} exceeds "
                f"{fmt.fmt_size(CAT_MAX_SIZE)} limit"
            )
        with tempfile.TemporaryDirectory(prefix="hf-bsh-cat-") as tmp:
            local = Path(tmp) / "f"
            st.client.download_files(st.bucket_id, [(rel, local)])
            data = local.read_bytes()
        if len(data) > CAT_MAX_SIZE:
            _bail(f"cat: {rel}: exceeds {fmt.fmt_size(CAT_MAX_SIZE)} limit")
        if b"\x00" in data[:8192]:
            print(f"cat: {rel}: binary file (skipped)")
            return
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        if data and not data.endswith(b"\n"):
            print()

    def do_du(self, arg: str):
        st = self.state
        if not st.is_opened():
            print("nothing opened")
            return
        tokens = _split_args(arg)
        human = "-h" in tokens
        rest = [t for t in tokens if t != "-h"]
        if len(rest) > 1:
            _bail("du: too many arguments (usage: du [-h] [path])")
        target = rest[0] if rest else ""
        total = 0
        for e in resolve_entries(st, target):
            if e.is_dir:
                total += sum(
                    x.size or 0 for x in st.iter_tree(e.path, True) if not x.is_dir
                )
            else:
                total += e.size or 0
        print(fmt.fmt_size(total) if human else total)

    def do_find(self, arg: str):
        st = self.state
        if not st.is_opened():
            print("nothing opened")
            return
        for e in resolve_entries(st, arg.strip()):
            if e.is_dir:
                for x in st.iter_tree(e.path, True):
                    print(x.path)
            else:
                print(e.path)

    def do_tree(self, arg: str):
        st = self.state
        if not st.is_opened():
            print("nothing opened")
            return
        tokens = _split_args(arg)
        max_depth: Optional[int] = None
        rest: List[str] = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "-L" and i + 1 < len(tokens):
                try:
                    max_depth = int(tokens[i + 1])
                except ValueError:
                    _bail("invalid depth")
                i += 2
            else:
                rest.append(tokens[i])
                i += 1
        arg0 = rest[0] if rest else ""
        if has_glob_chars(arg0):
            targets = [e.path for e in st.glob_match(arg0)]
        else:
            targets = [st.remote_prefix(arg0)]
        multi = len(targets) > 1
        for idx, rel in enumerate(targets):
            if multi and idx > 0:
                print()
            _tree_one(st, rel, max_depth)

    def do_rm(self, arg: str):
        self.state.ensure_opened("rm")
        tokens = _split_args(arg)
        recursive = "-r" in tokens
        args = [t for t in tokens if t != "-r"]
        if not args:
            _bail("rm: missing path (usage: rm [-r] <path>...)")
        st = self.state
        to_delete: List[str] = []
        for a in args:
            for rel in st.resolve_targets(a):
                if recursive:
                    files = st.expand_recursive(rel)
                    to_delete.extend(files if files else [rel])
                else:
                    to_delete.append(rel)
        if not to_delete:
            return
        st.client.batch(st.bucket_id, delete=to_delete)
        for p in to_delete:
            print(f"removed {p}")
        st.invalidate_cache()

    def do_mv(self, arg: str):
        self._move_or_copy(arg, delete_sources=True)

    def do_cp(self, arg: str):
        self._move_or_copy(arg, delete_sources=False)

    def _move_or_copy(self, arg: str, delete_sources: bool):
        op = "mv" if delete_sources else "cp"
        tokens = _split_args(arg)
        if len(tokens) < 2:
            _bail(f"{op}: usage: {op} <src>... <dst>")
        srcs_args, dst_arg = tokens[:-1], tokens[-1]
        st = self.state
        st.ensure_opened(op)
        bucket = st.bucket_id

        # Resolve sources, keeping is_dir + origin so dirs can be expanded and
        # external files routed to copy_files.
        # Each entry: (origin, kind, repo_id, path, is_dir)
        src_entries = []
        for s in srcs_args:
            url = parse_hf_url(s)
            if url is not None:
                if delete_sources:
                    _bail(f"{op}: external sources (hf://...) can only be copied, not moved")
                if has_glob_chars(url.path):
                    _bail(f"{op}: globs in hf:// sources aren't supported yet (use concrete paths)")
                if not url.path:
                    _bail(f"{op}: missing path in {s!r}")
                src_entries.append((_Origin.EXTERNAL, url.kind, url.repo_id, url.path, False))
            elif has_glob_chars(s):
                for e in st.glob_match(s):
                    src_entries.append((_Origin.OWN, None, None, e.path, e.is_dir))
            else:
                rel = st.remote_prefix(s)
                if rel == "":
                    _bail(f"{op}: cannot use root as a source")
                stt = st.stat(rel)
                if stt is None:
                    _bail(f"{op}: source not found: {rel}")
                src_entries.append((_Origin.OWN, None, None, rel, stt[0]))
        if not src_entries:
            _bail(f"{op}: no sources")

        dst_is_dir = dst_arg.endswith("/")
        if not dst_is_dir:
            s = st.stat(st.remote_prefix(dst_arg))
            dst_is_dir = bool(s and s[0])
        if len(src_entries) > 1 and not dst_is_dir:
            _bail(
                f"{op}: target {dst_arg!r} is not a directory "
                "(add trailing / or use an existing directory)"
            )
        dst_base = st.remote_prefix(dst_arg.rstrip("/"))

        # Expand to (origin, kind, repo_id, src_file, dst_file).
        pairs = []
        for origin, kind, repo_id, path, is_dir in src_entries:
            if dst_is_dir:
                leaf = basename(path)
                landing = f"{dst_base}/{leaf}" if dst_base else leaf
            else:
                landing = dst_base
            if is_dir:
                prefix = f"{path}/"
                saw_file = False
                for e in st.iter_tree(path, True):
                    if e.is_dir:
                        continue
                    saw_file = True
                    sub = e.path[len(prefix):] if e.path.startswith(prefix) else e.path
                    dst = f"{landing}/{sub}" if landing else sub
                    pairs.append((origin, kind, repo_id, e.path, dst))
                if not saw_file:
                    _bail(f"{op}: directory {path!r} is empty")
            else:
                pairs.append((origin, kind, repo_id, path, landing))

        # Own-bucket copies go through one batch (server-side xet copy by hash);
        # external sources go through copy_files (it resolves the hash itself).
        own = [(s, d) for (o, _k, _r, s, d) in pairs if o == _Origin.OWN]
        own_copies = []
        if own:
            hashes = {e.path: e.xet_hash for e in st.client.paths_info(bucket, [s for s, _ in own])}
            for s, d in own:
                xh = hashes.get(s)
                if xh is None:
                    _bail(f"{op}: source not found: {s}")
                own_copies.append((KIND_BUCKET, bucket, xh, d))
        deletes = [s for s, _ in own] if delete_sources else None

        for origin, kind, repo_id, s, d in pairs:
            if origin == _Origin.EXTERNAL:
                src_uri = st.client.repo_uri(kind, repo_id, s)
                dst_uri = f"hf://buckets/{bucket}/{d}"
                st.client.copy_files(src_uri, dst_uri)
        st.client.batch(bucket, copy=own_copies, delete=deletes)

        verb = "moved" if delete_sources else "copied"
        for origin, kind, repo_id, s, d in pairs:
            if origin == _Origin.OWN:
                print(f"{verb} {s} -> {d}")
            else:
                print(f"copied hf://{kind_segment(kind)}/{repo_id}/{s} -> {d}")
        st.invalidate_cache()

    def do_put(self, arg: str):
        st = self.state
        st.ensure_opened("put")
        tokens = _split_args(arg)
        if len(tokens) < 2:
            _bail("put: usage: put <local-src>... <remote-dst>")
        srcs_args, dst_arg = tokens[:-1], tokens[-1]

        sources: List[Path] = []
        for s in srcs_args:
            s_fs = expand_tilde(s)
            if has_glob_chars(s_fs):
                matched = globmod.glob(s_fs)
                if not matched:
                    _bail(f"put: no match: {s}")
                sources.extend(Path(m) for m in matched)
            else:
                p = Path(s_fs)
                if not p.exists():
                    _bail(f"put: local source not found: {p}")
                sources.append(p)
        if not sources:
            _bail("put: no sources")

        dst_rel = st.remote_prefix(dst_arg.rstrip("/"))
        dst_is_dir = (
            dst_arg.endswith("/")
            or bool((lambda s: s and s[0])(st.stat(dst_rel)))
            or len(sources) > 1
            or any(p.is_dir() for p in sources)
        )
        if len(sources) > 1 and not dst_is_dir:
            _bail(f"put: target {dst_arg!r} is not a directory")

        pairs: List[Tuple[Path, str]] = []
        for src in sources:
            if dst_is_dir:
                leaf = src.name
                if not leaf:
                    _bail(f"put: cannot use {src} as a source")
                landing = f"{dst_rel}/{leaf}" if dst_rel else leaf
            else:
                landing = dst_rel
            if src.is_dir():
                saw_file = False
                for file, sub in _walk_local_dir(src):
                    saw_file = True
                    dst = f"{landing}/{sub}" if landing else sub
                    pairs.append((file, dst))
                if not saw_file:
                    _bail(f"put: directory {str(src)!r} is empty")
            elif src.is_file():
                pairs.append((src, landing))
            else:
                _bail(f"put: unsupported source type: {src}")

        add = [(local, remote) for local, remote in pairs]
        n = len(add)
        print(f"uploading {n} file{'' if n == 1 else 's'}…", file=sys.stderr)
        st.client.batch(st.bucket_id, add=add)
        for local, remote in pairs:
            size = local.stat().st_size if local.exists() else 0
            print(f"uploaded {local} -> {remote} ({size} bytes)")
        st.invalidate_cache()

    def do_get(self, arg: str):
        st = self.state
        st.ensure_opened("get")
        tokens = _split_args(arg)
        if not tokens:
            _bail("get: usage: get <remote-src>... [<local-dst>]")
        if len(tokens) == 1:
            srcs_args, dst_arg = tokens, "."
        else:
            srcs_args, dst_arg = tokens[:-1], tokens[-1]

        src_entries: List[Tuple[str, bool]] = []
        for s in srcs_args:
            if has_glob_chars(s):
                for e in st.glob_match(s):
                    src_entries.append((e.path, e.is_dir))
            else:
                rel = st.remote_prefix(s)
                stt = st.stat(rel)
                if stt is None:
                    _bail(f"get: remote source not found: {rel}")
                src_entries.append((rel, stt[0]))
        if not src_entries:
            _bail("get: no sources")

        dst_path = Path(expand_tilde(dst_arg))
        dst_is_dir = (
            dst_arg.endswith("/")
            or dst_arg in (".", "..")
            or dst_path.is_dir()
            or len(src_entries) > 1
            or any(is_dir for _, is_dir in src_entries)
        )
        if len(src_entries) > 1 and not dst_is_dir:
            _bail(f"get: target {dst_arg!r} is not a directory")

        pairs: List[Tuple[str, Path]] = []
        for src_path, is_dir in src_entries:
            if dst_is_dir:
                leaf = basename(src_path)
                if not leaf:
                    _bail(f"get: cannot infer filename for {src_path!r}")
                landing = dst_path / leaf
            else:
                landing = dst_path
            if is_dir:
                prefix = f"{src_path}/"
                saw_file = False
                for e in st.iter_tree(src_path, True):
                    if e.is_dir:
                        continue
                    saw_file = True
                    sub = e.path[len(prefix):] if e.path.startswith(prefix) else e.path
                    pairs.append((e.path, landing / Path(*sub.split("/"))))
                if not saw_file:
                    _bail(f"get: directory {src_path!r} is empty")
            else:
                pairs.append((src_path, landing))

        # Resolve sizes for the summary; download in one call.
        infos = {e.path: e for e in st.client.paths_info(st.bucket_id, [r for r, _ in pairs])}
        n = len(pairs)
        print(f"downloading {n} file{'' if n == 1 else 's'}…", file=sys.stderr)
        st.client.download_files(st.bucket_id, [(rem, loc) for rem, loc in pairs])
        for rem, loc in pairs:
            size = infos[rem].size if rem in infos and infos[rem].size is not None else 0
            print(f"downloaded {rem} -> {loc} ({size} bytes)")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _split_args(arg: str) -> List[str]:
    try:
        return shlex.split(arg)
    except ValueError as e:
        _bail(f"parse error: {e}")


def _walk_local_dir(root: Path):
    """Yield `(file_path, posix_rel_path)` for every regular file under `root`.
    `posix_rel_path` uses `/` separators regardless of platform."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            fp = Path(dirpath) / fn
            if fp.is_file() and not fp.is_symlink():
                rel = fp.relative_to(root).as_posix()
                yield fp, rel


class _TreeNode:
    __slots__ = ("is_dir", "size", "mtime", "children")

    def __init__(self):
        self.is_dir = False
        self.size = None
        self.mtime = None
        self.children: Dict[str, "_TreeNode"] = {}


def _tree_one(st: State, rel: str, max_depth: Optional[int]):
    base = f"{rel.rstrip('/')}/" if rel else ""
    root_label = st.url_root() if not rel else rel
    root = _TreeNode()
    n_dirs = 0
    n_files = 0
    for e in st.iter_tree(rel, True):
        rel_path = (e.path[len(base):] if base and e.path.startswith(base) else e.path).strip("/")
        if not rel_path:
            continue
        parts = rel_path.split("/")
        if max_depth is not None and len(parts) > max_depth:
            node = root
            for depth, p in enumerate(parts, start=1):
                if depth > max_depth:
                    break
                if p not in node.children:
                    node.children[p] = _TreeNode()
                    node.children[p].is_dir = True
                    n_dirs += 1
                node = node.children[p]
            continue
        node = root
        for p in parts[:-1]:
            if p not in node.children:
                node.children[p] = _TreeNode()
                node.children[p].is_dir = True
                n_dirs += 1
            node = node.children[p]
        leaf = parts[-1]
        leaf_size = None if e.is_dir else e.size
        if leaf not in node.children:
            node.children[leaf] = _TreeNode()
            if e.is_dir:
                n_dirs += 1
            else:
                n_files += 1
        existing = node.children[leaf]
        if existing.is_dir or e.is_dir:
            existing.is_dir = True
        if existing.size is None:
            existing.size = leaf_size
        if existing.mtime is None:
            existing.mtime = e.mtime

    print(root_label)
    _walk_tree(root, "")
    print(f"\n{n_dirs} directories, {n_files} files")


def _walk_tree(node: _TreeNode, prefix_str: str):
    entries = sorted(node.children.items(), key=lambda kv: (not kv[1].is_dir, kv[0]))
    n = len(entries)
    for i, (name, child) in enumerate(entries):
        last = i == n - 1
        branch = "└── " if last else "├── "
        size_s = "" if child.is_dir else fmt.fmt_size(child.size)
        time_s = fmt.fmt_mtime(child.mtime)
        suffix = "/" if child.is_dir else ""
        print(f"{time_s:>12}  {size_s:>10}  {prefix_str}{branch}{name}{suffix}")
        if child.children:
            next_prefix = prefix_str + ("    " if last else "│   ")
            _walk_tree(child, next_prefix)


def print_help():
    print("commands:")
    print("  open <ns>/<name>             open a bucket (read/write)")
    print("  cd <path> | cd .. | cd /     change dir (. / .. / / / absolute paths OK)")
    print("  ls [path]                    list")
    print("  pwd                          print hf:// URL")
    print("  cat <path>                   dump a text file (<=1 MiB)")
    print("  du [-h] [path]               total bytes (-h: human-readable)")
    print("  find [path]                  recursive path dump")
    print("  tree [-L N] [path]           tree view")
    print("  rm [-r] <path>…              delete (bucket only)")
    print("  mv <src>... <dst>            move files/dirs within the bucket")
    print("  cp <src>... <dst>            copy files/dirs; <src> can be an hf://… URL")
    print("  put <local-src>... <dst>     upload local files/dirs into bucket")
    print("  get <remote-src>... [<dst>]  download remote files/dirs to local fs (default: .)")
    print("  refresh                      clear completion cache")
    print("  exit | quit                  leave the shell")
    print()
    print("paths support glob patterns (*, ?, [..]) in the final component,")
    print("  e.g. `rm -r checkpoint-*`, `mv data/*.parquet archive/`")
