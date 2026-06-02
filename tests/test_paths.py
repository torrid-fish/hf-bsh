import os

import pytest

from hf_bsh.paths import (
    basename,
    expand_tilde,
    has_glob_chars,
    normalize_path,
    parse_hf_url,
    split_cmd,
    KIND_BUCKET,
    KIND_DATASET,
    KIND_MODEL,
)


def test_split_cmd_basic():
    assert split_cmd("ls") == ("ls", "")
    assert split_cmd("ls foo") == ("ls", "foo")
    assert split_cmd("  ls   foo bar") == ("ls", "foo bar")
    assert split_cmd("cat path/with/spaces in it") == ("cat", "path/with/spaces in it")


def test_has_glob_chars():
    assert has_glob_chars("*.txt")
    assert has_glob_chars("foo?")
    assert has_glob_chars("file[0-9].bin")
    assert has_glob_chars("dir/*.md")
    assert not has_glob_chars("foo")
    assert not has_glob_chars("a/b/c.txt")
    assert not has_glob_chars("")


def test_basename():
    assert basename("foo/bar/baz.txt") == "baz.txt"
    assert basename("standalone") == "standalone"
    assert basename("") == ""


def test_normalize_path():
    assert normalize_path("") == ""
    assert normalize_path("a/b") == "a/b"
    assert normalize_path("a/./b") == "a/b"
    assert normalize_path("a/../b") == "b"
    assert normalize_path("a/b/..") == "a"
    assert normalize_path("a//b") == "a/b"
    # clamps at root
    assert normalize_path("..") == ""
    assert normalize_path("../x") == "x"
    assert normalize_path("a/../../b") == "b"


def test_expand_tilde():
    home = os.path.expanduser("~")
    if home == "~":
        pytest.skip("no home dir")
    home = home.rstrip("/")
    assert expand_tilde("~") == home
    assert expand_tilde("~/x/y") == f"{home}/x/y"
    assert expand_tilde("foo/bar") == "foo/bar"
    assert expand_tilde("./~/y") == "./~/y"
    assert expand_tilde("~user/x") == "~user/x"


def test_parse_hf_url_bucket():
    u = parse_hf_url("hf://buckets/alice/my-bucket/path/to/file")
    assert (u.kind, u.repo_id, u.path) == (KIND_BUCKET, "alice/my-bucket", "path/to/file")
    u = parse_hf_url("hf://buckets/alice/b")
    assert (u.repo_id, u.path) == ("alice/b", "")
    with pytest.raises(ValueError):
        parse_hf_url("hf://buckets/alice")
    with pytest.raises(ValueError):
        parse_hf_url("hf://buckets//")


def test_parse_hf_url_dataset_legacy_single_segment():
    u = parse_hf_url("hf://datasets/squad/train.parquet")
    assert (u.kind, u.repo_id, u.path) == (KIND_DATASET, "squad", "train.parquet")


def test_parse_hf_url_dataset_modern_two_segment_nested():
    u = parse_hf_url("hf://datasets/HuggingFaceH4/zephyr-7b/data/train.parquet")
    assert (u.kind, u.repo_id, u.path) == (KIND_DATASET, "HuggingFaceH4/zephyr-7b", "data/train.parquet")


def test_parse_hf_url_model_two_segment():
    u = parse_hf_url("hf://models/meta-llama/Llama-3.1-8B/config.json")
    assert (u.kind, u.repo_id, u.path) == (KIND_MODEL, "meta-llama/Llama-3.1-8B", "config.json")


def test_parse_hf_url_rejects_and_passthrough():
    assert parse_hf_url("s3://bucket/x") is None
    assert parse_hf_url("relative/path") is None
    with pytest.raises(ValueError):
        parse_hf_url("hf://spaces/x/y")
