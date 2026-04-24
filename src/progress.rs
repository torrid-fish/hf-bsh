//! Minimal TTY progress bar for `put` / `get`.
//!
//! `ProgressBar::new` spawns a render thread that repaints `\r{line}` on stderr
//! every 100 ms; `ProgressBar::finish` stops it and prints a final newline.
//! When stderr is not a TTY the renderer is suppressed and `finish` prints a
//! single summary line — safe to pipe or redirect.
//!
//! Counters live behind atomics so polling threads (e.g. the xet progress
//! poller in `api.rs`) can update them without locks. Obtain a [`ProgressHandle`]
//! via [`ProgressBar::handle`] and hand it to the api layer.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::fmt::fmt_size;

pub struct ProgressBar {
    inner: Arc<Inner>,
    render: Option<JoinHandle<()>>,
}

/// Cheap, thread-safe writer-side view of a [`ProgressBar`]. Multiple handles
/// can coexist — they all mutate the same atomic counters.
#[derive(Clone)]
pub struct ProgressHandle {
    inner: Arc<Inner>,
}

struct Inner {
    prefix: String,
    start: Instant,
    total: AtomicU64,      // 0 = unknown (server didn't send Content-Length)
    completed: AtomicU64,
    stop: AtomicBool,
    tty: bool,
    last_line_len: Mutex<usize>,
}

impl ProgressBar {
    /// Start a new bar. `total` may be 0 if the size is unknown up-front —
    /// call [`ProgressHandle::set_total`] later once it's learned.
    pub fn new(prefix: impl Into<String>, total: u64) -> Self {
        let inner = Arc::new(Inner {
            prefix: prefix.into(),
            start: Instant::now(),
            total: AtomicU64::new(total),
            completed: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            tty: std::io::stderr().is_terminal(),
            last_line_len: Mutex::new(0),
        });
        let render = if inner.tty {
            let i = Arc::clone(&inner);
            Some(thread::spawn(move || render_loop(i)))
        } else {
            None
        };
        Self { inner, render }
    }

    pub fn handle(&self) -> ProgressHandle {
        ProgressHandle { inner: Arc::clone(&self.inner) }
    }

    /// Stop the render thread and emit a final summary line.
    pub fn finish(mut self) {
        self.inner.stop.store(true, Ordering::Release);
        if let Some(h) = self.render.take() {
            let _ = h.join();
        }
        let completed = self.inner.completed.load(Ordering::Acquire);
        let total = self.inner.total.load(Ordering::Acquire);
        let line = format_line(&self.inner, completed, total);
        let mut stderr = std::io::stderr().lock();
        if self.inner.tty {
            let last = self
                .inner
                .last_line_len
                .lock()
                .ok()
                .map(|m| *m)
                .unwrap_or(0);
            let pad = last.saturating_sub(line.chars().count());
            let _ = write!(stderr, "\r{}{}\n", line, " ".repeat(pad));
        } else {
            let _ = writeln!(stderr, "{}", line);
        }
        let _ = stderr.flush();
    }
}

impl Drop for ProgressBar {
    fn drop(&mut self) {
        // If the caller didn't invoke finish() (e.g. on an error path), at
        // least stop the renderer so it doesn't keep scribbling on stderr.
        self.inner.stop.store(true, Ordering::Release);
        if let Some(h) = self.render.take() {
            let _ = h.join();
        }
    }
}

impl ProgressHandle {
    pub fn set_completed(&self, n: u64) {
        self.inner.completed.store(n, Ordering::Release);
    }
    /// Streaming callers (chunked HTTP reads, incremental uploads) bump the
    /// completed counter one chunk at a time. Currently unused because all
    /// transfers run through xet's own snapshot-style `progress()` reports,
    /// but kept as part of the handle's public API for future use.
    #[allow(dead_code)]
    pub fn add(&self, n: u64) {
        self.inner.completed.fetch_add(n, Ordering::Release);
    }
    pub fn set_total(&self, n: u64) {
        self.inner.total.store(n, Ordering::Release);
    }
}

fn render_loop(inner: Arc<Inner>) {
    while !inner.stop.load(Ordering::Acquire) {
        let completed = inner.completed.load(Ordering::Acquire);
        let total = inner.total.load(Ordering::Acquire);
        let line = format_line(&inner, completed, total);
        let mut stderr = std::io::stderr().lock();
        let mut last = inner.last_line_len.lock().expect("last_line_len poisoned");
        let pad = last.saturating_sub(line.chars().count());
        let _ = write!(stderr, "\r{}{}", line, " ".repeat(pad));
        let _ = stderr.flush();
        *last = line.chars().count();
        drop(last);
        drop(stderr);
        thread::sleep(Duration::from_millis(100));
    }
}

fn format_line(inner: &Inner, completed: u64, total: u64) -> String {
    let elapsed = inner.start.elapsed().as_secs_f64().max(0.001);
    let rate_bps = (completed as f64 / elapsed) as u64;
    let rate_s = format!("{:>10}/s", fmt_size(Some(rate_bps)));
    let done_s = fmt_size(Some(completed));
    if total > 0 {
        let pct = (completed as f64 / total as f64 * 100.0).min(100.0);
        let total_s = fmt_size(Some(total));
        format!(
            "{}  {:>5.1}%  {:>10} / {:>10}  {}",
            inner.prefix, pct, done_s, total_s, rate_s
        )
    } else {
        format!("{}         {:>10}  {}", inner.prefix, done_s, rate_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_line_with_known_total() {
        let inner = Inner {
            prefix: "uploading".into(),
            start: Instant::now(),
            total: AtomicU64::new(1000),
            completed: AtomicU64::new(250),
            stop: AtomicBool::new(false),
            tty: false,
            last_line_len: Mutex::new(0),
        };
        let line = format_line(&inner, 250, 1000);
        assert!(line.contains("uploading"));
        assert!(line.contains("25.0%"));
    }

    #[test]
    fn format_line_unknown_total_drops_percentage() {
        let inner = Inner {
            prefix: "downloading".into(),
            start: Instant::now(),
            total: AtomicU64::new(0),
            completed: AtomicU64::new(42),
            stop: AtomicBool::new(false),
            tty: false,
            last_line_len: Mutex::new(0),
        };
        let line = format_line(&inner, 42, 0);
        assert!(line.contains("downloading"));
        assert!(!line.contains("%"));
    }

    #[test]
    fn finish_without_tty_prints_newline_and_drops_cleanly() {
        // Smoke: just make sure finish() returns and the drop path doesn't panic.
        let bar = ProgressBar::new("test", 10);
        bar.handle().set_completed(5);
        bar.finish();
    }
}
