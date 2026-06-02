"""Offline shell tests driven by an in-memory fake client — exercises the
command logic (path resolution, glob, dir expansion, cp/mv/get planning)
without touching the network."""

from pathlib import Path

import pytest

from hf_bsh.api import TreeEntry
from hf_bsh.shell import Shell, ShellError


class FakeClient:
    """Emulates the bucket tree from a flat dict of {path: size}."""

    def __init__(self, files):
        self.files = dict(files)
        self.batch_calls = []
        self.copy_calls = []
        self.downloaded = []

    # tree emulation -----------------------------------------------------
    def list_bucket_tree(self, bucket_id, prefix, recursive):
        base = prefix or ""
        matching = [f for f in self.files if f.startswith(base)]
        if recursive:
            return [TreeEntry(f, False, self.files[f], None, f"hash:{f}") for f in matching]
        children = {}
        for f in matching:
            rest = f[len(base):]
            if "/" in rest:
                d = base + rest.split("/", 1)[0]
                children.setdefault(d, TreeEntry(d, True, None, None))
            else:
                children[f] = TreeEntry(f, False, self.files[f], None, f"hash:{f}")
        return list(children.values())

    def paths_info(self, bucket_id, paths):
        return [
            TreeEntry(p, False, self.files[p], None, f"hash:{p}")
            for p in paths
            if p in self.files
        ]

    def batch(self, bucket_id, add=None, copy=None, delete=None):
        self.batch_calls.append({"add": add, "copy": copy, "delete": delete})
        # reflect deletes/adds into the in-memory tree for follow-up assertions
        for d in delete or []:
            self.files.pop(d, None)
        for _src, dst in add or []:
            self.files[dst] = 1

    def copy_files(self, source, destination):
        self.copy_calls.append((source, destination))

    def download_files(self, bucket_id, files):
        for remote, local in files:
            self.downloaded.append((remote, str(local)))
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            Path(local).write_text(f"content:{remote}\n")

    def list_buckets(self, namespace=None):
        return ["alice/data", "alice/models"]

    def list_datasets(self, search, limit):
        return ["squad"]

    def list_models(self, search, limit):
        return ["meta-llama/Llama-3.1-8B"]

    def repo_uri(self, kind, repo_id, path):
        from hf_bsh.paths import kind_segment

        return f"hf://{kind_segment(kind)}/{repo_id}/{path}"


SAMPLE = {
    "config.json": 10,
    "weights.bin": 2048,
    "checkpoints/ckpt-1.bin": 100,
    "checkpoints/ckpt-2.bin": 200,
    "data/train.parquet": 500,
}


def mk(files=SAMPLE):
    sh = Shell(FakeClient(files))
    return sh


def test_open_accepts_bare_and_buckets_prefix():
    sh = mk()
    sh.run_line("open alice/my-bucket")
    assert sh.state.bucket_id == "alice/my-bucket"
    sh = mk()
    sh.run_line("open buckets/alice/my-bucket")
    assert sh.state.bucket_id == "alice/my-bucket"


def test_open_rejects_bad_targets():
    sh = mk()
    for bad in [
        "open",
        "open alice",
        "open buckets/alice",
        "open alice/my-bucket/extra",
        "open datasets/squad",
        "open models/meta-llama/Llama-3.1-8B",
        "open hf://buckets/alice/my-bucket",
    ]:
        with pytest.raises(ShellError):
            sh.run_line(bad)


def test_cd_pwd_and_normalize(capsys):
    sh = mk()
    sh.run_line("open alice/data")
    sh.run_line("cd checkpoints")
    assert sh.state.cwd == "checkpoints"
    sh.run_line("cd ../data")
    assert sh.state.cwd == "data"
    sh.run_line("cd /")
    assert sh.state.cwd == ""
    sh.run_line("pwd")
    assert capsys.readouterr().out.strip() == "hf://buckets/alice/data"


def test_ls_dirs_first(capsys):
    sh = mk()
    sh.run_line("open alice/data")
    sh.run_line("ls")
    lines = capsys.readouterr().out.strip().splitlines()
    names = [ln.split()[-1] for ln in lines]
    assert names[:2] == ["checkpoints/", "data/"]
    assert "config.json" in names and "weights.bin" in names


def test_du_recursive(capsys):
    sh = mk()
    sh.run_line("open alice/data")
    sh.run_line("du")
    assert capsys.readouterr().out.strip() == str(10 + 2048 + 100 + 200 + 500)
    sh.run_line("du -h checkpoints")
    assert capsys.readouterr().out.strip() == "300 B"


def test_find(capsys):
    sh = mk()
    sh.run_line("open alice/data")
    sh.run_line("find checkpoints")
    out = set(capsys.readouterr().out.split())
    assert out == {"checkpoints/ckpt-1.bin", "checkpoints/ckpt-2.bin"}


def test_glob_ls_and_no_match():
    sh = mk()
    sh.run_line("open alice/data")
    # *.json matches config.json
    sh.run_line("ls *.json")
    with pytest.raises(ShellError):
        sh.run_line("ls *.nope")


def test_rm_recursive_expands(capsys):
    sh = mk()
    sh.run_line("open alice/data")
    sh.run_line("rm -r checkpoints")
    deletes = sh.state.client.batch_calls[-1]["delete"]
    assert set(deletes) == {"checkpoints/ckpt-1.bin", "checkpoints/ckpt-2.bin"}


def test_cp_own_bucket_uses_batch_copy():
    sh = mk()
    sh.run_line("open alice/data")
    sh.run_line("cp config.json backup.json")
    copy = sh.state.client.batch_calls[-1]["copy"]
    assert copy == [("bucket", "alice/data", "hash:config.json", "backup.json")]


def test_mv_copies_then_deletes():
    sh = mk()
    sh.run_line("open alice/data")
    sh.run_line("mv config.json renamed.json")
    call = sh.state.client.batch_calls[-1]
    assert call["copy"] == [("bucket", "alice/data", "hash:config.json", "renamed.json")]
    assert call["delete"] == ["config.json"]


def test_cp_external_hf_url_uses_copy_files():
    sh = mk()
    sh.run_line("open alice/data")
    sh.run_line("cp hf://datasets/squad/train.parquet raw/")
    assert sh.state.client.copy_calls == [
        ("hf://datasets/squad/train.parquet", "hf://buckets/alice/data/raw/train.parquet")
    ]


def test_mv_external_rejected():
    sh = mk()
    sh.run_line("open alice/data")
    with pytest.raises(ShellError):
        sh.run_line("mv hf://datasets/squad/train.parquet raw/")


def test_get_expands_dir(tmp_path):
    sh = mk()
    sh.run_line("open alice/data")
    sh.run_line(f"get checkpoints {tmp_path}")
    got = {r for r, _ in sh.state.client.downloaded}
    assert got == {"checkpoints/ckpt-1.bin", "checkpoints/ckpt-2.bin"}
    assert (tmp_path / "checkpoints" / "ckpt-1.bin").exists()


def test_unknown_command():
    sh = mk()
    with pytest.raises(ShellError):
        sh.run_line("frobnicate x")


def test_exit_returns_true():
    sh = mk()
    assert sh.run_line("exit") is True
    assert sh.run_line("quit") is True
    assert sh.run_line("ls") is False
