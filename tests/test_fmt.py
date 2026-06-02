from hf_bsh.fmt import fmt_entry, fmt_size, parse_datetime


def test_fmt_size_none_is_empty():
    assert fmt_size(None) == ""


def test_fmt_size_bytes_no_decimal():
    assert fmt_size(512) == "512 B"


def test_fmt_size_unit_ladder():
    assert fmt_size(1024) == "1.0 KB"
    assert fmt_size(1024 * 1024) == "1.0 MB"
    assert fmt_size(1024 ** 3) == "1.0 GB"


def test_fmt_size_caps_at_pb():
    assert fmt_size(2 * 1024 ** 5).endswith(" PB")


def test_parse_datetime():
    assert parse_datetime("2024-06-01T12:34:56Z") is not None
    assert parse_datetime("2024-06-01T12:34:56.000Z") is not None
    assert parse_datetime("2024-06-01T12:34:56+00:00") is not None
    assert parse_datetime("not a date") is None


def test_fmt_entry_directory():
    s = fmt_entry(True, None, None, "subdir")
    assert s.endswith("subdir/")
    assert s.startswith("          ")  # blank size column


def test_fmt_entry_file_shows_size():
    s = fmt_entry(False, 1024, None, "f")
    assert "1.0 KB" in s
    assert s.endswith("f")
