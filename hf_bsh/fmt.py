"""Size / time / listing formatting. Ported from the Rust `fmt.rs` so `ls` and
`tree` output line up the same way."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Union

_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def fmt_size(n: Optional[int]) -> str:
    if n is None:
        return ""
    f = float(n)
    for i, u in enumerate(_UNITS):
        if f < 1024.0 or i == len(_UNITS) - 1:
            if u == "B":
                return f"{int(f)} {u}"
            return f"{f:.1f} {u}"
        f /= 1024.0
    raise AssertionError("unreachable")


def _coerce_dt(value: Union[None, str, datetime]) -> Optional[datetime]:
    """Accept whatever `huggingface_hub` hands back (a `datetime` or an ISO
    string) and normalise to an aware UTC `datetime`."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return parse_datetime(value)


def parse_datetime(s: str) -> Optional[datetime]:
    """Parse an ISO 8601 / RFC 3339 timestamp into an aware UTC `datetime`."""
    if not s:
        return None
    text = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fmt_mtime(value: Union[None, str, datetime]) -> str:
    dt = _coerce_dt(value)
    if dt is None:
        return ""
    local = dt.astimezone()
    now = datetime.now(local.tzinfo)
    age_days = (now - local).days
    future_days = (local - now).days
    if age_days >= 180 or future_days >= 1:
        return local.strftime("%b %d  %Y")
    return local.strftime("%b %d %H:%M")


def fmt_entry(
    is_dir: bool,
    size: Optional[int],
    mtime: Union[None, str, datetime],
    name: str,
) -> str:
    size_s = "" if is_dir else fmt_size(size)
    time_s = fmt_mtime(mtime)
    suffix = "/" if is_dir else ""
    return f"{time_s:>12}  {size_s:>10}  {name}{suffix}"
