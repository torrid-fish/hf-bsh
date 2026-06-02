"""Thin wrapper over `huggingface_hub.HfApi`.

The Rust port hand-rolled every REST call and the xet CAS data plane (~830
lines). `huggingface_hub` already ships all of that — bucket tree listing,
paths-info, the add/copy/delete batch endpoint, xet upload/download, and auth
token resolution (env vars + the on-disk token file). So this module is just a
small adapter that turns library objects into the plain `TreeEntry` shape the
shell layer works with."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from huggingface_hub import HfApi

from .paths import kind_segment


@dataclass
class TreeEntry:
    """One listing entry — file or directory."""

    path: str
    is_dir: bool
    size: Optional[int] = None
    mtime: Union[None, str, datetime] = None
    xet_hash: Optional[str] = None


def _to_entry(item) -> TreeEntry:
    is_dir = (
        getattr(item, "type", "") == "directory"
        or item.__class__.__name__ == "BucketFolder"
    )
    size = None if is_dir else getattr(item, "size", None)
    mtime = getattr(item, "mtime", None) or getattr(item, "uploaded_at", None)
    return TreeEntry(
        path=item.path,
        is_dir=is_dir,
        size=size,
        mtime=mtime,
        xet_hash=getattr(item, "xet_hash", None),
    )


class Client:
    """Auth + endpoint config and the handful of bucket operations the shell needs.

    `token=None` lets `huggingface_hub` resolve credentials itself (the
    `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` env vars, then the saved token file) —
    the same precedence the Rust version implemented by hand."""

    def __init__(self, endpoint: Optional[str] = None, token: Optional[str] = None):
        self.endpoint = endpoint or os.environ.get("HF_ENDPOINT") or None
        if self.endpoint:
            self.endpoint = self.endpoint.rstrip("/")
        self.token = token
        self.api = HfApi(endpoint=self.endpoint, token=token)

    # ---- identity / discovery (used by completion) ----

    def whoami_name(self) -> str:
        return self.api.whoami()["name"]

    def list_buckets(self, namespace: Optional[str] = None) -> List[str]:
        return [b.id for b in self.api.list_buckets(namespace or None)]

    def list_datasets(self, search: Optional[str], limit: int) -> List[str]:
        items = self.api.list_datasets(search=search or None, limit=limit)
        return [d.id for d in items]

    def list_models(self, search: Optional[str], limit: int) -> List[str]:
        items = self.api.list_models(search=search or None, limit=limit)
        return [m.id for m in items]

    # ---- bucket tree / metadata ----

    def list_bucket_tree(
        self, bucket_id: str, prefix: Optional[str], recursive: bool
    ) -> List[TreeEntry]:
        items = self.api.list_bucket_tree(
            bucket_id, prefix=prefix or None, recursive=recursive
        )
        return [_to_entry(it) for it in items]

    def paths_info(self, bucket_id: str, paths: Iterable[str]) -> List[TreeEntry]:
        paths = list(paths)
        if not paths:
            return []
        return [_to_entry(it) for it in self.api.get_bucket_paths_info(bucket_id, paths)]

    # ---- mutations ----

    def batch(
        self,
        bucket_id: str,
        add: Optional[List[Tuple[Union[str, Path, bytes], str]]] = None,
        copy: Optional[List[Tuple[str, str, str, str]]] = None,
        delete: Optional[List[str]] = None,
    ) -> None:
        if not (add or copy or delete):
            return
        self.api.batch_bucket_files(
            bucket_id, add=add or None, copy=copy or None, delete=delete or None
        )

    def copy_files(self, source: str, destination: str) -> None:
        """Server-side copy between hf:// locations (cross-repo or cross-bucket)."""
        self.api.copy_files(source, destination)

    def download_files(
        self, bucket_id: str, files: List[Tuple[str, Union[str, Path]]]
    ) -> None:
        if not files:
            return
        self.api.download_bucket_files(bucket_id, files=files)

    def repo_uri(self, kind: str, repo_id: str, path: str) -> str:
        """Build an `hf://` URI for a dataset/model/bucket source file."""
        return f"hf://{kind_segment(kind)}/{repo_id}/{path}"
