use chrono::{DateTime, Datelike, Local, TimeZone, Utc};

pub fn fmt_size(n: Option<u64>) -> String {
    let Some(n) = n else { return String::new() };
    let mut f = n as f64;
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    for (i, u) in UNITS.iter().enumerate() {
        if f < 1024.0 || i == UNITS.len() - 1 {
            if *u == "B" {
                return format!("{} {}", f as u64, u);
            }
            return format!("{:.1} {}", f, u);
        }
        f /= 1024.0;
    }
    unreachable!()
}

pub fn fmt_mtime(dt: Option<DateTime<Utc>>) -> String {
    let Some(dt) = dt else { return String::new() };
    let local = dt.with_timezone(&Local);
    let now = Local::now();
    let age = now.signed_duration_since(local);
    let future = local.signed_duration_since(now);
    if age.num_days() >= 180 || future.num_days() >= 1 {
        local.format("%b %d  %Y").to_string()
    } else {
        local.format("%b %d %H:%M").to_string()
    }
}

pub fn fmt_entry(
    is_dir: bool,
    size: Option<u64>,
    mtime: Option<DateTime<Utc>>,
    name: &str,
) -> String {
    let size_s = if is_dir {
        String::new()
    } else {
        fmt_size(size)
    };
    let time_s = fmt_mtime(mtime);
    let suffix = if is_dir { "/" } else { "" };
    format!("{:>12}  {:>10}  {}{}", time_s, size_s, name, suffix)
}

/// Parse an ISO 8601 / RFC 3339 timestamp into a `DateTime<Utc>`.
pub fn parse_datetime(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|| {
            // Some HF endpoints emit `YYYY-MM-DDTHH:MM:SS.fffZ` which rfc3339 accepts,
            // but fall back to a looser parse for bare `YYYY-MM-DDTHH:MM:SSZ`.
            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ")
                .ok()
                .map(|ndt| Utc.from_utc_datetime(&ndt))
        })
}

#[allow(dead_code)] // kept for parity with Datelike import; silences unused-warning on some toolchains
fn _use_datelike(dt: DateTime<Local>) -> i32 {
    dt.year()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_size_none_is_empty() {
        assert_eq!(fmt_size(None), "");
    }

    #[test]
    fn fmt_size_bytes_no_decimal() {
        assert_eq!(fmt_size(Some(512)), "512 B");
    }

    #[test]
    fn fmt_size_uses_unit_ladder() {
        assert_eq!(fmt_size(Some(1024)), "1.0 KB");
        assert_eq!(fmt_size(Some(1024 * 1024)), "1.0 MB");
        assert_eq!(fmt_size(Some(1024u64.pow(3))), "1.0 GB");
    }

    #[test]
    fn fmt_size_caps_at_pb() {
        // 2 PiB — should still render in PB
        let n = 2u64 * 1024u64.pow(5);
        assert!(fmt_size(Some(n)).ends_with(" PB"));
    }

    #[test]
    fn parse_datetime_rfc3339() {
        assert!(parse_datetime("2024-06-01T12:34:56Z").is_some());
        assert!(parse_datetime("2024-06-01T12:34:56.000Z").is_some());
        assert!(parse_datetime("2024-06-01T12:34:56+00:00").is_some());
    }

    #[test]
    fn parse_datetime_bogus() {
        assert!(parse_datetime("not a date").is_none());
    }

    #[test]
    fn fmt_entry_directory_has_trailing_slash_and_empty_size() {
        let s = fmt_entry(true, None, None, "subdir");
        assert!(s.ends_with("subdir/"));
        // leading 10 cols of size are blank
        assert!(s.starts_with("          "));
    }

    #[test]
    fn fmt_entry_file_shows_size() {
        let s = fmt_entry(false, Some(1024), None, "f");
        assert!(s.contains("1.0 KB"));
        assert!(s.ends_with("f"));
    }
}
