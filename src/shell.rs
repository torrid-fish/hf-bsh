use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;

use anyhow::{anyhow, bail, Context as _, Result};
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Helper};

use crate::api::{BucketAddOp, BucketCopyOp, BucketPathInfo, Client, RepoKind, TreeEntry};
use crate::fmt::{fmt_entry, fmt_size};
use crate::progress::ProgressBar;

pub const CAT_MAX_SIZE: u64 = 1 << 20; // 1 MiB

/// Mutable shell state. Held inside an `Rc<RefCell<_>>` so the rustyline helper
/// and the REPL loop can both touch it. `bucket_id` being empty means "no
/// bucket currently opened" — we're bucket-only now, so a simple string is
/// enough (no repo kind enum).
pub struct State {
    pub client: Client,
    pub bucket_id: String,
    pub cwd: String,
    pub ls_cache: HashMap<String, Vec<String>>,
}

impl State {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            bucket_id: String::new(),
            cwd: String::new(),
            ls_cache: HashMap::new(),
        }
    }

    fn is_opened(&self) -> bool {
        !self.bucket_id.is_empty()
    }

    fn ensure_opened(&self, op: &str) -> Result<()> {
        if self.is_opened() {
            Ok(())
        } else {
            bail!("{}: nothing opened", op)
        }
    }

    fn url_root(&self) -> String {
        if self.is_opened() {
            format!("hf://buckets/{}", self.bucket_id)
        } else {
            "hf://".to_string()
        }
    }

    pub fn prompt(&self) -> String {
        if !self.is_opened() {
            return "hf> ".to_string();
        }
        let shown = if self.cwd.is_empty() { String::new() } else { format!("/{}", self.cwd) };
        format!("hf:{}{}> ", self.bucket_id, shown)
    }

    fn invalidate_cache(&mut self) {
        self.ls_cache.clear();
    }

    fn remote_prefix(&self, p: &str) -> String {
        // Absolute paths jump to root; everything else joins under cwd.
        let raw = if let Some(rest) = p.strip_prefix('/') {
            rest.to_string()
        } else if p.is_empty() {
            self.cwd.clone()
        } else if self.cwd.is_empty() {
            p.to_string()
        } else {
            format!("{}/{}", self.cwd, p)
        };
        normalize_path(&raw)
    }

    fn iter_tree(&self, rel_dir: &str, recursive: bool) -> Result<Vec<TreeEntry>> {
        if !self.is_opened() {
            bail!("nothing opened");
        }
        let prefix = if rel_dir.is_empty() { None } else { Some(rel_dir) };
        let prefix_owned = prefix.map(|p| format!("{}/", p.trim_end_matches('/')));
        self.client
            .list_bucket_tree(&self.bucket_id, prefix_owned.as_deref(), recursive)
    }

    /// Return immediate children of `rel_dir` as raw names (trailing `/` on dirs). Cached.
    fn listdir(&mut self, rel_dir: &str) -> Vec<String> {
        if !self.is_opened() {
            return Vec::new();
        }
        if let Some(c) = self.ls_cache.get(rel_dir) {
            return c.clone();
        }
        let base = if rel_dir.is_empty() {
            String::new()
        } else {
            format!("{}/", rel_dir.trim_end_matches('/'))
        };
        let mut names = Vec::new();
        match self.iter_tree(rel_dir, false) {
            Ok(entries) => {
                for e in entries {
                    let tail = if !base.is_empty() && e.path.starts_with(&base) {
                        &e.path[base.len()..]
                    } else {
                        &e.path[..]
                    };
                    let tail = tail.trim_matches('/');
                    if tail.is_empty() {
                        continue;
                    }
                    names.push(if e.is_dir { format!("{}/", tail) } else { tail.to_string() });
                }
            }
            Err(_) => return Vec::new(),
        }
        self.ls_cache.insert(rel_dir.to_string(), names.clone());
        names
    }

    fn stat(&self, rel: &str) -> Result<Option<(bool, Option<u64>)>> {
        // Root ("") is always a directory — it never appears in its own
        // parent listing, so the generic path below would return None.
        if rel.is_empty() {
            return Ok(Some((true, None)));
        }
        let parent = rel.rsplit_once('/').map(|(p, _)| p).unwrap_or("").to_string();
        let entries = self.iter_tree(&parent, false)?;
        for e in entries {
            if e.path == rel {
                return Ok(Some((e.is_dir, e.size)));
            }
        }
        Ok(None)
    }

    /// Expand `rel` to all file paths underneath it. If `rel` is itself a file,
    /// returns `[rel]`. If it doesn't exist or contains no files, returns empty.
    fn expand_recursive(&self, rel: &str) -> Result<Vec<String>> {
        Ok(self
            .iter_tree(rel, true)?
            .into_iter()
            .filter(|e| !e.is_dir)
            .map(|e| e.path)
            .collect())
    }

    /// Match `arg` (which may contain `*`, `?`, or `[...]`) against the entries
    /// of its literal parent directory. Glob characters may appear **only in the
    /// final path component**. Errors with "no match" (zsh-style) if nothing
    /// matches. Caller should only invoke this when `has_glob_chars(arg)`.
    fn glob_match(&self, arg: &str) -> Result<Vec<TreeEntry>> {
        let (dir_part, leaf_pat) = match arg.rsplit_once('/') {
            Some((d, p)) => (d, p),
            None => ("", arg),
        };
        if has_glob_chars(dir_part) {
            bail!(
                "glob: patterns only supported in the final path component (got {:?})",
                arg
            );
        }
        let base_rel = self.remote_prefix(dir_part);
        let pat = glob::Pattern::new(leaf_pat)
            .map_err(|e| anyhow!("bad pattern {:?}: {}", leaf_pat, e))?;
        let prefix = if base_rel.is_empty() {
            String::new()
        } else {
            format!("{}/", base_rel)
        };
        let mut out = Vec::new();
        for e in self.iter_tree(&base_rel, false)? {
            let leaf = e
                .path
                .strip_prefix(&prefix)
                .unwrap_or(&e.path)
                .trim_matches('/');
            if !leaf.is_empty() && pat.matches(leaf) {
                out.push(e);
            }
        }
        if out.is_empty() {
            bail!("no match: {}", arg);
        }
        Ok(out)
    }

    /// One-shot "arg → absolute paths" resolver used by rm / find / tree / du.
    /// - No glob chars: returns `[remote_prefix(arg)]` regardless of existence.
    /// - With glob chars: returns all matches, errors on zero matches.
    fn resolve_targets(&self, arg: &str) -> Result<Vec<String>> {
        if has_glob_chars(arg) {
            Ok(self.glob_match(arg)?.into_iter().map(|e| e.path).collect())
        } else {
            Ok(vec![self.remote_prefix(arg)])
        }
    }
}

/// Resolve `arg` into a list of [`TreeEntry`] values rooted under `st.cwd`,
/// handling the three cases uniformly: glob expansion, empty-arg (= current
/// directory), and a single literal path. Non-glob literal paths are stat'd so
/// callers can distinguish files from directories without blindly calling
/// `iter_tree` on a file (which would 404 on the `/tree/` endpoint).
pub(crate) fn resolve_entries(st: &State, arg: &str) -> Result<Vec<TreeEntry>> {
    if has_glob_chars(arg) {
        return st.glob_match(arg);
    }
    let rel = st.remote_prefix(arg);
    if rel.is_empty() {
        return Ok(vec![TreeEntry { path: String::new(), is_dir: true, size: None, mtime: None }]);
    }
    match st.stat(&rel)? {
        Some((is_dir, size)) => Ok(vec![TreeEntry { path: rel, is_dir, size, mtime: None }]),
        None => bail!("not found: {}", rel),
    }
}

/// Collapse `.` and `..` segments in a slash-joined path, drop empty segments.
/// Works on already-joined strings — leading `/` is not preserved (the caller
/// passes a repo-relative path anyway). `..` that would escape root clamps to
/// root (empty string), matching `cd ../../..` behavior at shallow depths.
pub(crate) fn normalize_path(p: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    out.join("/")
}

pub(crate) fn has_glob_chars(s: &str) -> bool {
    s.chars().any(|c| matches!(c, '*' | '?' | '['))
}

/// Return the final path component of `p` (no trailing slash handling — callers
/// should strip first if they care).
pub(crate) fn basename(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

/// Where a `cp` / `mv` source physically lives. Originating repo dictates both
/// how we fetch the xet hash and what `sourceRepoType` / `sourceRepoId` to send
/// in the `/batch` NDJSON.
#[derive(Debug, Clone)]
enum SourceOrigin {
    OpenedBucket,
    External {
        kind: RepoKind,
        repo_id: String,
    },
}

#[derive(Debug, Clone)]
struct ResolvedSource {
    origin: SourceOrigin,
    path: String,
    is_dir: bool,
}

/// A parsed `hf://{buckets,datasets,models}/<repo>/<path>` URL. `repo_id` is
/// `<ns>/<name>` for buckets (always namespaced) or whatever the dataset/model
/// id is otherwise. `path` is the file path inside that repo (never starts with
/// `/`).
#[derive(Debug)]
pub(crate) struct HfUrl {
    pub kind: RepoKind,
    pub repo_id: String,
    pub path: String,
}

/// Parse `hf://{buckets,datasets,models}/<id>/<path>`. Returns `Ok(None)` when
/// `s` isn't an `hf://` URL (caller should treat it as a bucket-relative path),
/// `Err` when it is an `hf://` URL but malformed.
pub(crate) fn parse_hf_url(s: &str) -> Result<Option<HfUrl>> {
    let Some(rest) = s.strip_prefix("hf://") else {
        return Ok(None);
    };
    let (kind, after_prefix) = if let Some(r) = rest.strip_prefix("buckets/") {
        (RepoKind::Bucket, r)
    } else if let Some(r) = rest.strip_prefix("datasets/") {
        (RepoKind::Dataset, r)
    } else if let Some(r) = rest.strip_prefix("models/") {
        (RepoKind::Model, r)
    } else {
        bail!("hf url: must start with hf://{{buckets,datasets,models}}/ (got {:?})", s);
    };
    // Buckets need `<ns>/<name>/<path>` — two slash-separated id components
    // before the path. Datasets/models may be single-segment ids.
    let (repo_id, path) = match kind {
        RepoKind::Bucket => {
            let mut it = after_prefix.splitn(3, '/');
            let ns = it.next().unwrap_or("");
            let name = it.next().unwrap_or("");
            let path = it.next().unwrap_or("");
            if ns.is_empty() || name.is_empty() {
                bail!("hf url: bucket id must be <ns>/<name> (got {:?})", s);
            }
            (format!("{}/{}", ns, name), path.to_string())
        }
        RepoKind::Dataset | RepoKind::Model => {
            // Dataset/model ids are either `<name>` (legacy, e.g. `squad`) or
            // `<ns>/<name>` (modern, e.g. `HuggingFaceH4/zephyr-7b`). Given
            // `after_prefix = <id>/<path>`:
            //   - ≥2 slashes → id is the first two segments (modern form).
            //   - exactly 1 slash → id is the first segment (legacy form).
            // This gets single-segment-id-with-nested-path wrong (rare edge
            // case — e.g. a `squad/nested/file.txt` would be misread as
            // id=`squad/nested`), but that shape isn't in common use on the hub.
            let slashes = after_prefix.matches('/').count();
            match slashes {
                0 => bail!("hf url: missing <path> after repo id (got {:?})", s),
                1 => {
                    let (id, path) = after_prefix.split_once('/').unwrap();
                    if id.is_empty() || path.is_empty() {
                        bail!("hf url: empty repo id or path (got {:?})", s);
                    }
                    (id.to_string(), path.to_string())
                }
                _ => {
                    let mut it = after_prefix.splitn(3, '/');
                    let ns = it.next().unwrap_or("");
                    let name = it.next().unwrap_or("");
                    let path = it.next().unwrap_or("");
                    if ns.is_empty() || name.is_empty() || path.is_empty() {
                        bail!("hf url: empty segment in {:?}", s);
                    }
                    (format!("{}/{}", ns, name), path.to_string())
                }
            }
        }
    };
    Ok(Some(HfUrl { kind, repo_id, path }))
}

// ----------------------------------------------------------------------
// Public shell dispatch: executes one line of REPL input.
// ----------------------------------------------------------------------

pub struct Shell {
    pub state: Rc<RefCell<State>>,
}

impl Shell {
    pub fn new(client: Client) -> Self {
        Self {
            state: Rc::new(RefCell::new(State::new(client))),
        }
    }

    pub fn prompt(&self) -> String {
        self.state.borrow().prompt()
    }

    /// Run one line. Returns `Ok(true)` if the shell should exit.
    pub fn run_line(&mut self, line: &str) -> Result<bool> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(false);
        }
        let (cmd, rest) = split_cmd(line);
        match cmd {
            "open" => self.do_open(rest)?,
            "cd" => self.do_cd(rest)?,
            "pwd" => self.do_pwd(),
            "ls" => self.do_ls(rest)?,
            "cat" => self.do_cat(rest)?,
            "du" => self.do_du(rest)?,
            "find" => self.do_find(rest)?,
            "tree" => self.do_tree(rest)?,
            "rm" => self.do_rm(rest)?,
            "mv" => self.do_mv(rest)?,
            "cp" => self.do_cp(rest)?,
            "put" => self.do_put(rest)?,
            "get" => self.do_get(rest)?,
            "refresh" => {
                self.state.borrow_mut().invalidate_cache();
                println!("cache cleared");
            }
            "help" | "?" => print_help(),
            "exit" | "quit" => return Ok(true),
            other => bail!("unknown command: {}", other),
        }
        Ok(false)
    }

    // ------------ commands ------------

    fn do_open(&mut self, arg: &str) -> Result<()> {
        let arg = arg.trim();
        if arg.is_empty() {
            bail!("usage: open buckets/<ns>/<name>");
        }
        let rest = arg.strip_prefix("buckets/").ok_or_else(|| {
            anyhow!(
                "open: only buckets can be opened (got {:?}); to pull from a dataset/model, \
                 use `cp hf://datasets/<id>/<path> <dst>` from an opened bucket",
                arg
            )
        })?;
        if rest.is_empty() {
            bail!("open: missing bucket id after `buckets/` (got {:?})", arg);
        }
        if !rest.contains('/') {
            bail!("open: buckets require <ns>/<name> (got {:?})", arg);
        }
        let mut st = self.state.borrow_mut();
        st.bucket_id = rest.to_string();
        st.cwd = String::new();
        st.invalidate_cache();
        Ok(())
    }

    fn do_cd(&mut self, arg: &str) -> Result<()> {
        let arg = arg.trim().trim_end_matches('/');
        let new_cwd = if arg.is_empty() {
            String::new()
        } else {
            self.state.borrow().remote_prefix(arg)
        };
        let mut st = self.state.borrow_mut();
        st.cwd = new_cwd;
        st.invalidate_cache();
        Ok(())
    }

    fn do_pwd(&self) {
        let st = self.state.borrow();
        let full = format!("{}/{}", st.url_root(), st.cwd);
        println!("{}", full.trim_end_matches('/'));
    }

    fn do_ls(&self, arg: &str) -> Result<()> {
        let st = self.state.borrow();
        if !st.is_opened() {
            println!("nothing opened");
            return Ok(());
        }
        let arg = arg.trim();

        // Glob path: print each match one line at a time, showing its leaf name.
        if has_glob_chars(arg) {
            let mut matches = st.glob_match(arg)?;
            matches.sort_by(|a, b| match (b.is_dir, a.is_dir) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => a.path.cmp(&b.path),
            });
            for e in matches {
                let leaf = basename(e.path.trim_end_matches('/')).to_string();
                println!("{}", fmt_entry(e.is_dir, e.size, e.mtime, &leaf));
            }
            return Ok(());
        }

        // Non-glob path: list inside the directory (as before).
        let rel = st.remote_prefix(arg);
        let base = if rel.is_empty() {
            String::new()
        } else {
            format!("{}/", rel.trim_end_matches('/'))
        };
        let mut items: Vec<(String, bool, Option<u64>, Option<chrono::DateTime<chrono::Utc>>)> = Vec::new();
        for e in st.iter_tree(&rel, false)? {
            let tail = if !base.is_empty() && e.path.starts_with(&base) {
                &e.path[base.len()..]
            } else {
                &e.path[..]
            };
            let tail = tail.trim_matches('/').to_string();
            if tail.is_empty() {
                continue;
            }
            items.push((tail, e.is_dir, e.size, e.mtime));
        }
        items.sort_by(|a, b| match (b.1, a.1) {
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => a.0.cmp(&b.0),
        });
        for (name, is_dir, size, mtime) in items {
            println!("{}", fmt_entry(is_dir, size, mtime, &name));
        }
        Ok(())
    }

    fn do_cat(&self, arg: &str) -> Result<()> {
        let st = self.state.borrow();
        if !st.is_opened() {
            println!("nothing opened");
            return Ok(());
        }
        let arg = arg.trim();
        if arg.is_empty() {
            bail!("cat: missing path");
        }
        // Glob: require a single match (concatenating under a 1 MiB cap is messy).
        let rel = if has_glob_chars(arg) {
            let matches = st.glob_match(arg)?;
            if matches.len() != 1 {
                bail!(
                    "cat: pattern {:?} matches {} entries; only single-file cat supported",
                    arg,
                    matches.len()
                );
            }
            matches.into_iter().next().unwrap().path
        } else {
            st.remote_prefix(arg)
        };
        let display = if rel.is_empty() { "/" } else { rel.as_str() };
        let Some((is_dir, size)) = st.stat(&rel)? else {
            bail!("cat: {}: not found", display);
        };
        if is_dir {
            bail!("cat: {}: is a directory", display);
        }
        if let Some(n) = size {
            if n > CAT_MAX_SIZE {
                bail!(
                    "cat: {}: {} exceeds {} limit",
                    rel,
                    fmt_size(Some(n)),
                    fmt_size(Some(CAT_MAX_SIZE))
                );
            }
        }
        st.ensure_opened("cat")?;
        // Single-match cat only — batch concatenation with a 1 MiB cap is a
        // foot-gun (see CLAUDE.md).
        let info = st
            .client
            .bucket_paths_info(&st.bucket_id, std::slice::from_ref(&rel))?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("cat: {}: not found", rel))?;
        let data = st
            .client
            .download_bucket_file(&st.bucket_id, &info, CAT_MAX_SIZE)?;
        if data.len() as u64 > CAT_MAX_SIZE {
            bail!("cat: {}: exceeds {} limit", rel, fmt_size(Some(CAT_MAX_SIZE)));
        }
        let head_len = data.len().min(8192);
        if data[..head_len].contains(&0u8) {
            println!("cat: {}: binary file (skipped)", rel);
            return Ok(());
        }
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&data)?;
        stdout.flush()?;
        if !data.is_empty() && !data.ends_with(b"\n") {
            println!();
        }
        Ok(())
    }

    fn do_du(&self, arg: &str) -> Result<()> {
        let st = self.state.borrow();
        if !st.is_opened() {
            println!("nothing opened");
            return Ok(());
        }
        let tokens = shell_words::split(arg).map_err(|e| anyhow!("parse error: {}", e))?;
        let human = tokens.iter().any(|t| t == "-h");
        let rest: Vec<&str> = tokens.iter().filter(|t| *t != "-h").map(String::as_str).collect();
        if rest.len() > 1 {
            bail!("du: too many arguments (usage: du [-h] [path])");
        }
        let target = rest.first().copied().unwrap_or("");
        let mut total: u64 = 0;
        for e in resolve_entries(&st, target)? {
            if e.is_dir {
                total += st
                    .iter_tree(&e.path, true)?
                    .into_iter()
                    .filter(|x| !x.is_dir)
                    .filter_map(|x| x.size)
                    .sum::<u64>();
            } else {
                total += e.size.unwrap_or(0);
            }
        }
        if human {
            println!("{}", fmt_size(Some(total)));
        } else {
            println!("{}", total);
        }
        Ok(())
    }

    fn do_find(&self, arg: &str) -> Result<()> {
        let st = self.state.borrow();
        if !st.is_opened() {
            println!("nothing opened");
            return Ok(());
        }
        for e in resolve_entries(&st, arg.trim())? {
            if e.is_dir {
                for x in st.iter_tree(&e.path, true)? {
                    println!("{}", x.path);
                }
            } else {
                println!("{}", e.path);
            }
        }
        Ok(())
    }

    fn do_tree(&self, arg: &str) -> Result<()> {
        let st = self.state.borrow();
        if !st.is_opened() {
            println!("nothing opened");
            return Ok(());
        }
        let tokens = shell_words::split(arg).map_err(|e| anyhow!("parse error: {}", e))?;
        let mut max_depth: Option<usize> = None;
        let mut rest: Vec<String> = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if tokens[i] == "-L" && i + 1 < tokens.len() {
                max_depth = Some(
                    tokens[i + 1]
                        .parse()
                        .map_err(|_| anyhow!("invalid depth"))?,
                );
                i += 2;
            } else {
                rest.push(tokens[i].clone());
                i += 1;
            }
        }
        let arg0 = rest.first().map(String::as_str).unwrap_or("");
        let targets: Vec<String> = if has_glob_chars(arg0) {
            st.glob_match(arg0)?.into_iter().map(|e| e.path).collect()
        } else {
            vec![st.remote_prefix(arg0)]
        };
        let multi = targets.len() > 1;
        for (i, rel) in targets.iter().enumerate() {
            if multi && i > 0 {
                println!();
            }
            Self::tree_one(&st, rel, max_depth)?;
        }
        Ok(())
    }

    /// Render a single `tree` root. Factored out so `do_tree` can iterate over
    /// glob matches and reuse it unchanged.
    fn tree_one(st: &State, rel: &str, max_depth: Option<usize>) -> Result<()> {
        let base = if rel.is_empty() {
            String::new()
        } else {
            format!("{}/", rel.trim_end_matches('/'))
        };
        let root_label = if rel.is_empty() {
            st.url_root()
        } else {
            rel.to_string()
        };

        // Build a trie of the paths.
        let mut root = TreeNode::default();
        let mut n_dirs: usize = 0;
        let mut n_files: usize = 0;
        for e in st.iter_tree(rel, true)? {
            let rel_path = if !base.is_empty() && e.path.starts_with(&base) {
                &e.path[base.len()..]
            } else {
                &e.path[..]
            };
            let rel_path = rel_path.trim_matches('/');
            if rel_path.is_empty() {
                continue;
            }
            let parts: Vec<&str> = rel_path.split('/').collect();
            let leaf_depth = parts.len();
            if let Some(md) = max_depth {
                if leaf_depth > md {
                    // still need to materialize the intermediate dirs up to md
                    let mut node = &mut root;
                    for (d, p) in parts.iter().enumerate() {
                        let depth = d + 1;
                        if depth > md {
                            break;
                        }
                        let entry = node.children.entry(p.to_string()).or_insert_with(|| {
                            n_dirs += 1;
                            TreeNode { is_dir: true, ..Default::default() }
                        });
                        node = entry;
                    }
                    continue;
                }
            }
            let mut node = &mut root;
            for p in &parts[..parts.len() - 1] {
                let entry = node.children.entry(p.to_string()).or_insert_with(|| {
                    n_dirs += 1;
                    TreeNode { is_dir: true, ..Default::default() }
                });
                node = entry;
            }
            let leaf = parts.last().unwrap().to_string();
            let leaf_size = if e.is_dir { None } else { e.size };
            let existing = node.children.entry(leaf).or_insert_with(|| {
                if e.is_dir {
                    n_dirs += 1;
                } else {
                    n_files += 1;
                }
                TreeNode::default()
            });
            if existing.is_dir || e.is_dir {
                existing.is_dir = true;
            }
            if existing.size.is_none() {
                existing.size = leaf_size;
            }
            if existing.mtime.is_none() {
                existing.mtime = e.mtime;
            }
        }

        println!("{}", root_label);
        walk_tree(&root, "");
        println!("\n{} directories, {} files", n_dirs, n_files);
        Ok(())
    }

    fn do_rm(&mut self, arg: &str) -> Result<()> {
        {
            let st = self.state.borrow();
            st.ensure_opened("rm")?;
        }
        let tokens = shell_words::split(arg).map_err(|e| anyhow!("parse error: {}", e))?;
        let recursive = tokens.iter().any(|t| t == "-r");
        let args: Vec<&str> = tokens.iter().filter(|t| *t != "-r").map(|s| s.as_str()).collect();
        if args.is_empty() {
            bail!("rm: missing path (usage: rm [-r] <path>...)");
        }

        let mut to_delete: Vec<String> = Vec::new();
        {
            let st = self.state.borrow();
            for a in &args {
                for rel in st.resolve_targets(a)? {
                    if recursive {
                        let files = st.expand_recursive(&rel)?;
                        if files.is_empty() {
                            to_delete.push(rel);
                        } else {
                            to_delete.extend(files);
                        }
                    } else {
                        to_delete.push(rel);
                    }
                }
            }
        }

        if to_delete.is_empty() {
            return Ok(());
        }
        let bucket = self.state.borrow().bucket_id.clone();
        self.state
            .borrow()
            .client
            .bucket_batch(&bucket, &[], &[], &to_delete)?;
        for p in &to_delete {
            println!("removed {}", p);
        }
        self.state.borrow_mut().invalidate_cache();
        Ok(())
    }

    fn do_mv(&mut self, arg: &str) -> Result<()> {
        self.do_move_or_copy(arg, true)
    }

    fn do_cp(&mut self, arg: &str) -> Result<()> {
        self.do_move_or_copy(arg, false)
    }

    /// Shared implementation for `mv` and `cp`. If `delete_sources` is true
    /// the source paths are also deleted in the same `/batch` call (= `mv`).
    ///
    /// Destination handling:
    /// - Trailing `/` in `dst`     → directory mode, each src lands at
    ///   `<dst>/<basename(src)>`.
    /// - `dst` is an existing dir  → directory mode (same as above).
    /// - Otherwise, single source  → rename: src lands at `<dst>`.
    /// - Otherwise, multiple srcs  → error (ambiguous).
    ///
    /// Directory sources are expanded recursively to their files; each file's
    /// sub-path relative to the source is appended to the landing base so the
    /// subtree structure is preserved.
    fn do_move_or_copy(&mut self, arg: &str, delete_sources: bool) -> Result<()> {
        let op = if delete_sources { "mv" } else { "cp" };
        let tokens = shell_words::split(arg).map_err(|e| anyhow!("parse error: {}", e))?;
        if tokens.len() < 2 {
            bail!("{}: usage: {} <src>... <dst>", op, op);
        }
        let (srcs_args, dst_arg) = tokens.split_at(tokens.len() - 1);
        let dst_arg = &dst_arg[0];
        {
            let st = self.state.borrow();
            st.ensure_opened(op)?;
        }

        // Resolve sources, keeping is_dir + origin info so we can expand dirs
        // and look up xet hashes from the right endpoint later. Sources fall
        // into three buckets: (a) relative paths inside the opened bucket,
        // (b) `hf://buckets/<other-ns>/<other>/path` external-bucket files,
        // (c) `hf://{datasets,models}/<id>/path` external-repo files.
        let bucket = self.state.borrow().bucket_id.clone();
        let mut src_entries: Vec<ResolvedSource> = Vec::new();
        for s in srcs_args {
            if let Some(url) = parse_hf_url(s)? {
                // External source. `mv` isn't allowed here — we don't own the
                // source-side delete permission for foreign repos.
                if delete_sources {
                    bail!("{}: external sources (hf://...) can only be copied, not moved", op);
                }
                if has_glob_chars(&url.path) {
                    bail!("{}: globs in hf:// sources aren't supported yet (use concrete paths)", op);
                }
                if url.path.is_empty() {
                    bail!("{}: missing path in {:?}", op, s);
                }
                // We don't attempt to recursively expand external dirs either;
                // that requires a second tree walk + per-file xet lookup.
                // Concrete file paths only, for now.
                src_entries.push(ResolvedSource {
                    origin: SourceOrigin::External {
                        kind: url.kind,
                        repo_id: url.repo_id,
                    },
                    path: url.path,
                    is_dir: false,
                });
            } else if has_glob_chars(s) {
                let st = self.state.borrow();
                for e in st.glob_match(s)? {
                    src_entries.push(ResolvedSource {
                        origin: SourceOrigin::OpenedBucket,
                        path: e.path,
                        is_dir: e.is_dir,
                    });
                }
            } else {
                let st = self.state.borrow();
                let rel = st.remote_prefix(s);
                if rel.is_empty() {
                    bail!("{}: cannot use root as a source", op);
                }
                match st.stat(&rel)? {
                    Some((is_dir, _)) => src_entries.push(ResolvedSource {
                        origin: SourceOrigin::OpenedBucket,
                        path: rel,
                        is_dir,
                    }),
                    None => bail!("{}: source not found: {}", op, rel),
                }
            }
        }
        if src_entries.is_empty() {
            bail!("{}: no sources", op);
        }

        // Decide dst semantics: trailing slash, or existing directory, or file.
        let dst_is_dir = dst_arg.ends_with('/') || {
            let rel = self.state.borrow().remote_prefix(dst_arg);
            match self.state.borrow().stat(&rel)? {
                Some((is_dir, _)) => is_dir,
                None => false,
            }
        };
        if src_entries.len() > 1 && !dst_is_dir {
            bail!(
                "{}: target {:?} is not a directory (add trailing / or use an existing directory)",
                op, dst_arg
            );
        }
        let dst_base = self
            .state
            .borrow()
            .remote_prefix(dst_arg.trim_end_matches('/'));

        // Expand each source (file or directory) to concrete (origin, src_file,
        // dst_file) triples. Own-bucket directories are walked so the subtree is
        // mirrored under the landing base; external-source dirs are rejected
        // upfront (see above).
        let mut pairs: Vec<(SourceOrigin, String, String)> = Vec::new();
        for entry in &src_entries {
            let landing = if dst_is_dir {
                let leaf = basename(&entry.path);
                if dst_base.is_empty() {
                    leaf.to_string()
                } else {
                    format!("{}/{}", dst_base, leaf)
                }
            } else {
                dst_base.clone()
            };
            if entry.is_dir {
                let prefix = format!("{}/", entry.path);
                let st = self.state.borrow();
                let mut saw_file = false;
                for e in st.iter_tree(&entry.path, true)? {
                    if e.is_dir {
                        continue;
                    }
                    saw_file = true;
                    let sub = e.path.strip_prefix(&prefix).unwrap_or(&e.path);
                    let dst = if landing.is_empty() {
                        sub.to_string()
                    } else {
                        format!("{}/{}", landing, sub)
                    };
                    pairs.push((entry.origin.clone(), e.path.clone(), dst));
                }
                if !saw_file {
                    bail!("{}: directory {:?} is empty", op, entry.path);
                }
            } else {
                pairs.push((entry.origin.clone(), entry.path.clone(), landing));
            }
        }

        // Fetch xet_hash per source. Group own-bucket paths into one paths-info
        // call; external files get one HEAD per file.
        let own_bucket_paths: Vec<String> = pairs
            .iter()
            .filter(|(o, _, _)| matches!(o, SourceOrigin::OpenedBucket))
            .map(|(_, s, _)| s.clone())
            .collect();
        let mut own_hashes: std::collections::HashMap<String, String> = if !own_bucket_paths.is_empty() {
            self.state
                .borrow()
                .client
                .bucket_paths_info(&bucket, &own_bucket_paths)?
                .into_iter()
                .map(|i| (i.path, i.xet_hash))
                .collect()
        } else {
            std::collections::HashMap::new()
        };

        let mut copies: Vec<BucketCopyOp> = Vec::with_capacity(pairs.len());
        for (origin, src, dst) in &pairs {
            match origin {
                SourceOrigin::OpenedBucket => {
                    let xet_hash = own_hashes
                        .remove(src)
                        .ok_or_else(|| anyhow!("{}: source not found: {}", op, src))?;
                    copies.push(BucketCopyOp {
                        source_repo_type: "bucket".into(),
                        source_repo_id: bucket.clone(),
                        xet_hash,
                        destination: dst.clone(),
                    });
                }
                SourceOrigin::External { kind, repo_id } => {
                    let info = match kind {
                        RepoKind::Bucket => self
                            .state
                            .borrow()
                            .client
                            .bucket_paths_info(repo_id, std::slice::from_ref(src))?
                            .into_iter()
                            .next()
                            .ok_or_else(|| anyhow!("{}: external source not found: {}", op, src))?,
                        RepoKind::Dataset | RepoKind::Model => self
                            .state
                            .borrow()
                            .client
                            .repo_xet_info(*kind, repo_id, src)?,
                    };
                    copies.push(BucketCopyOp {
                        source_repo_type: kind.repo_type_path().trim_end_matches('s').into(),
                        source_repo_id: repo_id.clone(),
                        xet_hash: info.xet_hash,
                        destination: dst.clone(),
                    });
                }
            }
        }

        let deletes: Vec<String> = if delete_sources {
            // Only own-bucket sources can be deleted; external ones were refused
            // upfront. Filter by origin so the check is local.
            pairs
                .iter()
                .filter(|(o, _, _)| matches!(o, SourceOrigin::OpenedBucket))
                .map(|(_, s, _)| s.clone())
                .collect()
        } else {
            Vec::new()
        };
        self.state
            .borrow()
            .client
            .bucket_batch(&bucket, &[], &copies, &deletes)?;
        let verb = if delete_sources { "moved" } else { "copied" };
        for (origin, src, dst) in &pairs {
            match origin {
                SourceOrigin::OpenedBucket => println!("{} {} -> {}", verb, src, dst),
                SourceOrigin::External { kind, repo_id } => println!(
                    "copied hf://{}/{}/{} -> {}",
                    kind.repo_type_path(),
                    repo_id,
                    src,
                    dst
                ),
            }
        }
        self.state.borrow_mut().invalidate_cache();
        Ok(())
    }

    /// `put <local-src>... <remote-dst>` — upload local files/dirs into the bucket.
    ///
    /// Destination rules mirror `cp` / `mv`:
    /// - Trailing `/`, existing remote dir, or >1 source → directory mode; each
    ///   source lands at `<dst>/<basename(src)>`.
    /// - Otherwise, single source → rename: src lands at `<dst>`.
    ///
    /// Local directories are walked recursively; the subtree is preserved under
    /// the landing base. Local globs (`*`, `?`, `[..]`) are expanded by the
    /// shell's glob crate and must match at least one file.
    fn do_put(&mut self, arg: &str) -> Result<()> {
        {
            let st = self.state.borrow();
            st.ensure_opened("put")?;
        }
        let tokens = shell_words::split(arg).map_err(|e| anyhow!("parse error: {}", e))?;
        if tokens.len() < 2 {
            bail!("put: usage: put <local-src>... <remote-dst>");
        }
        let (srcs_args, dst_arg) = tokens.split_at(tokens.len() - 1);
        let dst_arg = &dst_arg[0];

        // Expand local sources: each arg may be a literal file/dir or a glob.
        // `~` / `~/foo` is expanded to the user's home before globbing/stat.
        let mut sources: Vec<std::path::PathBuf> = Vec::new();
        for s in srcs_args {
            let expanded = expand_tilde(s);
            let s_fs = expanded.as_ref();
            if has_glob_chars(s_fs) {
                let mut matched = false;
                for entry in glob::glob(s_fs).map_err(|e| anyhow!("bad glob {:?}: {}", s, e))? {
                    let p = entry.map_err(|e| anyhow!("glob: {}", e))?;
                    matched = true;
                    sources.push(p);
                }
                if !matched {
                    bail!("put: no match: {}", s);
                }
            } else {
                let p = std::path::PathBuf::from(s_fs);
                if !p.exists() {
                    bail!("put: local source not found: {}", p.display());
                }
                sources.push(p);
            }
        }
        if sources.is_empty() {
            bail!("put: no sources");
        }

        // Decide dst semantics identical to mv/cp.
        let st_ref = self.state.borrow();
        let dst_rel = st_ref.remote_prefix(dst_arg.trim_end_matches('/'));
        let dst_is_dir = dst_arg.ends_with('/')
            || matches!(st_ref.stat(&dst_rel)?, Some((true, _)))
            || sources.len() > 1
            || sources.iter().any(|p| p.is_dir());
        drop(st_ref);
        if sources.len() > 1 && !dst_is_dir {
            bail!("put: target {:?} is not a directory", dst_arg);
        }

        // Build (local_path, remote_path) pairs, walking directories.
        let mut pairs: Vec<(std::path::PathBuf, String)> = Vec::new();
        for src in &sources {
            let landing = if dst_is_dir {
                let leaf = src
                    .file_name()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| anyhow!("put: cannot use {} as a source", src.display()))?;
                if dst_rel.is_empty() {
                    leaf.to_string()
                } else {
                    format!("{}/{}", dst_rel, leaf)
                }
            } else {
                dst_rel.clone()
            };
            if src.is_dir() {
                let mut saw_file = false;
                walk_local_dir(src, &mut |file, sub| {
                    saw_file = true;
                    let dst = if landing.is_empty() {
                        sub.to_string()
                    } else {
                        format!("{}/{}", landing, sub)
                    };
                    pairs.push((file.to_path_buf(), dst));
                })?;
                if !saw_file {
                    bail!("put: directory {:?} is empty", src.display().to_string());
                }
            } else if src.is_file() {
                pairs.push((src.clone(), landing));
            } else {
                bail!("put: unsupported source type: {}", src.display());
            }
        }

        let bucket = self.state.borrow().bucket_id.clone();
        let total_bytes: u64 = pairs
            .iter()
            .filter_map(|(p, _)| std::fs::metadata(p).ok().map(|m| m.len()))
            .sum();
        let bar = ProgressBar::new(
            format!("uploading {} file{}", pairs.len(), if pairs.len() == 1 { "" } else { "s" }),
            total_bytes,
        );
        let upload_result = self
            .state
            .borrow()
            .client
            .xet_upload_files(&bucket, &pairs, Some(bar.handle()));
        bar.finish();
        let uploaded = upload_result?;
        let mut adds: Vec<BucketAddOp> = Vec::with_capacity(uploaded.len());
        for (info, (local, _)) in uploaded.iter().zip(pairs.iter()) {
            let mtime_ms = std::fs::metadata(local)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as i64)
                .unwrap_or_else(|| {
                    (chrono::Utc::now().timestamp_millis()).max(0)
                });
            adds.push(BucketAddOp {
                destination: info.remote_path.clone(),
                xet_hash: info.xet_hash.clone(),
                mtime_ms,
                content_type: guess_content_type(&info.remote_path),
            });
        }
        self.state
            .borrow()
            .client
            .bucket_batch(&bucket, &adds, &[], &[])?;
        for ((local, _), info) in pairs.iter().zip(uploaded.iter()) {
            println!(
                "uploaded {} -> {} ({} bytes)",
                local.display(),
                info.remote_path,
                info.size
            );
        }
        self.state.borrow_mut().invalidate_cache();
        Ok(())
    }

    /// `get <remote-src>... [<local-dst>]` — download from the opened bucket
    /// to the local filesystem. `<local-dst>` defaults to `.`. Directory
    /// destinations (trailing `/`, existing dir, or >1 source) cause each
    /// source to land at `<dst>/<basename(src)>`; otherwise the single source
    /// is renamed to `<dst>`. Remote directories are expanded recursively and
    /// mirror their subtree locally.
    fn do_get(&mut self, arg: &str) -> Result<()> {
        {
            let st = self.state.borrow();
            st.ensure_opened("get")?;
        }
        let tokens = shell_words::split(arg).map_err(|e| anyhow!("parse error: {}", e))?;
        if tokens.is_empty() {
            bail!("get: usage: get <remote-src>... [<local-dst>]");
        }
        let (srcs_args, dst_arg): (Vec<String>, String) = if tokens.len() == 1 {
            (tokens.clone(), ".".to_string())
        } else {
            let last = tokens.last().unwrap().clone();
            (tokens[..tokens.len() - 1].to_vec(), last)
        };

        // Resolve remote sources (glob + stat), collecting (abs_remote_path, is_dir).
        let mut src_entries: Vec<(String, bool)> = Vec::new();
        {
            let st = self.state.borrow();
            for s in &srcs_args {
                if has_glob_chars(s) {
                    for e in st.glob_match(s)? {
                        src_entries.push((e.path, e.is_dir));
                    }
                } else {
                    let rel = st.remote_prefix(s);
                    match st.stat(&rel)? {
                        Some((is_dir, _)) => src_entries.push((rel, is_dir)),
                        None => bail!("get: remote source not found: {}", rel),
                    }
                }
            }
        }
        if src_entries.is_empty() {
            bail!("get: no sources");
        }

        let dst_path = std::path::PathBuf::from(expand_tilde(&dst_arg).as_ref());
        let dst_is_dir = dst_arg.ends_with('/')
            || dst_arg == "."
            || dst_arg == ".."
            || dst_path.is_dir()
            || src_entries.len() > 1
            || src_entries.iter().any(|(_, d)| *d);
        if src_entries.len() > 1 && !dst_is_dir {
            bail!("get: target {:?} is not a directory", dst_arg);
        }

        // Expand into (remote_file_rel, local_file_path) pairs.
        let mut pairs: Vec<(String, std::path::PathBuf)> = Vec::new();
        let st = self.state.borrow();
        for (src_path, is_dir) in &src_entries {
            let landing: std::path::PathBuf = if dst_is_dir {
                let leaf = basename(src_path);
                if leaf.is_empty() {
                    bail!("get: cannot infer filename for {:?}", src_path);
                }
                dst_path.join(leaf)
            } else {
                dst_path.clone()
            };
            if *is_dir {
                let prefix = format!("{}/", src_path);
                let mut saw_file = false;
                for e in st.iter_tree(src_path, true)? {
                    if e.is_dir {
                        continue;
                    }
                    saw_file = true;
                    let sub = e.path.strip_prefix(&prefix).unwrap_or(&e.path);
                    pairs.push((e.path.clone(), landing.join(sub.replace('/', std::path::MAIN_SEPARATOR_STR))));
                }
                if !saw_file {
                    bail!("get: directory {:?} is empty", src_path);
                }
            } else {
                pairs.push((src_path.clone(), landing));
            }
        }
        let bucket_id = st.bucket_id.clone();
        drop(st);

        // Resolve xet metadata for every remote file in one call.
        let remote_paths: Vec<String> = pairs.iter().map(|(r, _)| r.clone()).collect();
        let infos = self
            .state
            .borrow()
            .client
            .bucket_paths_info(&bucket_id, &remote_paths)?;
        let mut by_path: std::collections::HashMap<String, BucketPathInfo> =
            infos.into_iter().map(|i| (i.path.clone(), i)).collect();
        let mut downloads: Vec<(BucketPathInfo, std::path::PathBuf)> =
            Vec::with_capacity(pairs.len());
        for (rem, loc) in &pairs {
            let info = by_path
                .remove(rem)
                .ok_or_else(|| anyhow!("get: remote file not found: {}", rem))?;
            downloads.push((info, loc.clone()));
        }
        let total_bytes: u64 = downloads.iter().filter_map(|(i, _)| i.size).sum();
        let n = downloads.len();
        let bar = ProgressBar::new(
            format!("downloading {} file{}", n, if n == 1 { "" } else { "s" }),
            total_bytes,
        );
        let result = self.state.borrow().client.download_bucket_files_to_paths(
            &bucket_id,
            &downloads,
            Some(bar.handle()),
        );
        bar.finish();
        result?;
        for (info, loc) in &downloads {
            println!(
                "downloaded {} -> {} ({} bytes)",
                info.path,
                loc.display(),
                info.size.unwrap_or(0)
            );
        }
        Ok(())
    }
}

/// Walk `root` recursively, calling `cb(file_path, posix_rel_path)` for every
/// regular file encountered. `posix_rel_path` is relative to `root` and uses
/// `/` separators regardless of platform, ready for use as a remote path
/// segment.
fn walk_local_dir(
    root: &std::path::Path,
    cb: &mut dyn FnMut(&std::path::Path, &str),
) -> Result<()> {
    fn inner(
        root: &std::path::Path,
        dir: &std::path::Path,
        cb: &mut dyn FnMut(&std::path::Path, &str),
    ) -> Result<()> {
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("read_dir {}", dir.display()))?
        {
            let entry = entry.with_context(|| format!("read_dir entry in {}", dir.display()))?;
            let path = entry.path();
            let ft = entry.file_type()?;
            if ft.is_dir() {
                inner(root, &path, cb)?;
            } else if ft.is_file() {
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace(std::path::MAIN_SEPARATOR, "/");
                cb(&path, &rel);
            }
            // symlinks, sockets, etc. silently skipped.
        }
        Ok(())
    }
    inner(root, root, cb)
}

/// Cheap `mimetypes.guess_type`-style lookup for the handful of common types
/// a user actually uploads from the REPL. Returns `None` for unknowns; the
/// server fills in `application/octet-stream` by default.
fn guess_content_type(remote_path: &str) -> Option<String> {
    let ext = remote_path.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase())?;
    let ct = match ext.as_str() {
        "json" => "application/json",
        "txt" | "log" | "md" | "csv" | "tsv" => "text/plain",
        "html" | "htm" => "text/html",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "pdf" => "application/pdf",
        "gz" => "application/gzip",
        "zip" => "application/zip",
        "tar" => "application/x-tar",
        "parquet" => "application/vnd.apache.parquet",
        "safetensors" => "application/octet-stream",
        "bin" | "pt" | "pth" | "ckpt" | "onnx" => "application/octet-stream",
        _ => return None,
    };
    Some(ct.to_string())
}

// ----------------------------------------------------------------------
// Tree rendering
// ----------------------------------------------------------------------

#[derive(Default)]
struct TreeNode {
    is_dir: bool,
    size: Option<u64>,
    mtime: Option<chrono::DateTime<chrono::Utc>>,
    children: std::collections::BTreeMap<String, TreeNode>,
}

fn walk_tree(node: &TreeNode, prefix_str: &str) {
    let mut entries: Vec<(&String, &TreeNode)> = node.children.iter().collect();
    entries.sort_by(|a, b| match (b.1.is_dir, a.1.is_dir) {
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => a.0.cmp(b.0),
    });
    let n = entries.len();
    for (i, (name, child)) in entries.into_iter().enumerate() {
        let last = i == n - 1;
        let branch = if last { "└── " } else { "├── " };
        let size_s = if child.is_dir { String::new() } else { fmt_size(child.size) };
        let time_s = crate::fmt::fmt_mtime(child.mtime);
        let suffix = if child.is_dir { "/" } else { "" };
        println!(
            "{:>10}  {:>12}  {}{}{}{}",
            size_s, time_s, prefix_str, branch, name, suffix
        );
        if !child.children.is_empty() {
            let next_prefix = format!("{}{}", prefix_str, if last { "    " } else { "│   " });
            walk_tree(child, &next_prefix);
        }
    }
}

// ----------------------------------------------------------------------
// Path helpers
// ----------------------------------------------------------------------

pub(crate) fn split_cmd(line: &str) -> (&str, &str) {
    match line.find(|c: char| c.is_whitespace()) {
        Some(i) => (&line[..i], line[i..].trim_start()),
        None => (line, ""),
    }
}

fn print_help() {
    println!("commands:");
    println!("  open buckets/<ns>/<name>     open a bucket (read/write)");
    println!("  cd <path> | cd .. | cd /     change dir (. / .. / / / absolute paths OK)");
    println!("  ls [path]                    list");
    println!("  pwd                          print hf:// URL");
    println!("  cat <path>                   dump a text file (<=1 MiB)");
    println!("  du [-h] [path]               total bytes (-h: human-readable)");
    println!("  find [path]                  recursive path dump");
    println!("  tree [-L N] [path]           tree view");
    println!("  rm [-r] <path>…              delete (bucket only)");
    println!("  mv <src>... <dst>            move files/dirs within the bucket (dst must be dir for multi-src)");
    println!("  cp <src>... <dst>            copy files/dirs; <src> can be an hf://{{buckets,datasets,models}}/…");
    println!("                                 URL for server-side xet copy from another repo");
    println!("  put <local-src>... <dst>     upload local files/dirs into bucket");
    println!("  get <remote-src>... [<dst>]  download remote files/dirs to local fs (default: .)");
    println!("  refresh                      clear completion cache");
    println!("  exit | quit                  leave the shell");
    println!();
    println!("paths support glob patterns (*, ?, [..]) in the final component,");
    println!("  e.g. `rm -r checkpoint-*`, `mv data/*.parquet archive/`");
}

// ----------------------------------------------------------------------
// rustyline completer
// ----------------------------------------------------------------------

pub struct ShellHelper {
    pub state: Rc<RefCell<State>>,
}

impl Helper for ShellHelper {}
impl Hinter for ShellHelper {
    type Hint = String;
}
impl Highlighter for ShellHelper {}
impl Validator for ShellHelper {}

const COMMANDS: &[&str] = &[
    "open", "cd", "pwd", "ls", "cat", "du", "find", "tree", "rm", "mv", "cp", "put", "get",
    "refresh", "help", "exit", "quit",
];

impl Completer for ShellHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let before = &line[..pos];
        // Find the start of the current token (by whitespace only — slashes stay inside).
        let start = before
            .rfind(|c: char| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        let token = &before[start..];

        // Decide whether we're completing the command or an argument.
        let prefix = before[..start].trim_start();
        if prefix.is_empty() {
            let candidates: Vec<Pair> = COMMANDS
                .iter()
                .filter(|c| c.starts_with(token))
                .map(|c| Pair { display: (*c).to_string(), replacement: format!("{} ", c) })
                .collect();
            return Ok((start, candidates));
        }

        let cmd = prefix.split_whitespace().next().unwrap_or("");
        let candidates = match cmd {
            "open" => complete_open(&self.state, token),
            "cd" => {
                // Only directories.
                let mut all = complete_remote_path(&self.state, token);
                all.retain(|p| p.replacement.ends_with('/'));
                all
            }
            "ls" | "cat" | "du" | "find" | "tree" | "rm" => {
                complete_remote_path(&self.state, token)
            }
            "mv" | "cp" => {
                // `cp` / `mv` sources accept either a bucket-relative path or
                // an `hf://{buckets,datasets,models}/<id>[/<path>]` external URL.
                // Destinations are always bucket-relative, but while the user
                // is still typing we can't tell which token is "last", so both
                // kinds of candidates are offered (the relevant one wins).
                if token.starts_with("hf://") || "hf://".starts_with(token) {
                    complete_hf_url(&self.state, token)
                } else {
                    complete_remote_path(&self.state, token)
                }
            }
            "get" => {
                // `get <remote-src>... <local-dst>` — first positional arg is
                // the src (remote), subsequent ones favour the local dst. Can't
                // know for sure which token is "last" while the user is still
                // typing, so multi-src middle args will mis-complete; that's a
                // rarer workflow than the common `get <remote> <local>` pair.
                let arg_index = prefix.split_whitespace().count().saturating_sub(1);
                if arg_index == 0 {
                    complete_remote_path(&self.state, token)
                } else {
                    complete_local_path(token)
                }
            }
            "put" => complete_local_path(token),
            _ => Vec::new(),
        };
        Ok((start, candidates))
    }
}

fn complete_local_path(text: &str) -> Vec<Pair> {
    // Bare "~" is rare but convenient — offer "~/" as the single candidate so
    // the next keystroke starts listing $HOME.
    if text == "~" {
        return vec![Pair { display: "~/".into(), replacement: "~/".into() }];
    }
    let (dir_part, prefix) = match text.rsplit_once('/') {
        Some((d, p)) => (if d.is_empty() { "/" } else { d }, p),
        None => (".", text),
    };
    // Expand `~` for filesystem access but leave `text` (and therefore `head`
    // below) alone, so the replacement keeps the user's typed form.
    let fs_dir = expand_tilde(dir_part);
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(fs_dir.as_ref()) else {
        return out;
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(prefix) {
            continue;
        }
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let head = if text.contains('/') {
            let idx = text.rfind('/').unwrap() + 1;
            &text[..idx]
        } else {
            ""
        };
        let suffix = if is_dir { "/" } else { "" };
        let replacement = format!("{}{}{}", head, name, suffix);
        let display = format!("{}{}", name, suffix);
        out.push(Pair { display, replacement });
    }
    out
}

/// Expand a leading `~` / `~/` to the user's home directory. Everything else
/// is returned unchanged (borrowed, so no allocation on the common path).
/// `~user` is intentionally not supported — a REPL is unlikely to need it.
pub(crate) fn expand_tilde(s: &str) -> std::borrow::Cow<'_, str> {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            let home = home.to_string_lossy();
            return std::borrow::Cow::Owned(format!("{}/{}", home.trim_end_matches('/'), rest));
        }
    } else if s == "~" {
        if let Some(home) = dirs::home_dir() {
            return std::borrow::Cow::Owned(home.to_string_lossy().into_owned());
        }
    }
    std::borrow::Cow::Borrowed(s)
}

fn complete_remote_path(state: &Rc<RefCell<State>>, text: &str) -> Vec<Pair> {
    let (dir_part, prefix) = match text.rsplit_once('/') {
        Some((d, p)) => (d, p),
        None => ("", text),
    };
    let rel = {
        let st = state.borrow();
        st.remote_prefix(dir_part)
    };
    let names = state.borrow_mut().listdir(&rel);
    let head = if dir_part.is_empty() { String::new() } else { format!("{}/", dir_part) };
    names
        .into_iter()
        .filter(|n| n.starts_with(prefix))
        .map(|n| {
            let replacement = format!("{}{}", head, n);
            Pair { display: n, replacement }
        })
        .collect()
}

/// Tab-complete `hf://{buckets,datasets,models}/<id>` URLs for `cp` / `mv`
/// sources. We stop at the repo id — path-inside-repo completion would require
/// another API round-trip per keystroke and a cache, which isn't worth the
/// complexity right now. Users type the path after the id by hand.
fn complete_hf_url(state: &Rc<RefCell<State>>, text: &str) -> Vec<Pair> {
    let client = &state.borrow().client;
    let mut results: Vec<Pair> = Vec::new();

    // `hf://` itself — offer the three subtypes.
    if text == "hf://" || "hf://".starts_with(text) {
        let prefixes = ["hf://buckets/", "hf://datasets/", "hf://models/"];
        // If the user has typed a partial `hf://`, offer just the scheme; once
        // they complete `hf://` we can offer the three subtypes.
        if text.len() < "hf://".len() {
            results.push(Pair {
                display: "hf://".into(),
                replacement: "hf://".into(),
            });
            return results;
        }
        for p in prefixes {
            results.push(Pair { display: p.into(), replacement: p.into() });
        }
        return results;
    }

    // Past `hf://<kind>/` — complete repo/bucket ids.
    if let Some(q) = text.strip_prefix("hf://datasets/") {
        if let Ok(ids) = client.list_datasets(if q.is_empty() { None } else { Some(q) }, 30) {
            for id in ids {
                if id.starts_with(q) {
                    let s = format!("hf://datasets/{}/", id);
                    results.push(Pair { display: s.clone(), replacement: s });
                }
            }
        }
        return results;
    }
    if let Some(q) = text.strip_prefix("hf://models/") {
        if let Ok(ids) = client.list_models(if q.is_empty() { None } else { Some(q) }, 30) {
            for id in ids {
                if id.starts_with(q) {
                    let s = format!("hf://models/{}/", id);
                    results.push(Pair { display: s.clone(), replacement: s });
                }
            }
        }
        return results;
    }
    if let Some(q) = text.strip_prefix("hf://buckets/") {
        if let Ok(buckets) = client.list_buckets(None) {
            for b in buckets {
                if b.starts_with(q) {
                    let s = format!("hf://buckets/{}/", b);
                    results.push(Pair { display: s.clone(), replacement: s });
                }
            }
        }
        return results;
    }
    results
}

fn complete_open(state: &Rc<RefCell<State>>, text: &str) -> Vec<Pair> {
    let client = &state.borrow().client;
    let mut results: Vec<Pair> = Vec::new();
    if let Some(q) = text.strip_prefix("buckets/") {
        // Buckets in the user's namespace (requires a token).
        if let Ok(buckets) = client.list_buckets(None) {
            for b in buckets {
                if b.starts_with(q) {
                    let s = format!("buckets/{}", b);
                    results.push(Pair { display: s.clone(), replacement: s });
                }
            }
        }
        return results;
    }
    if "buckets/".starts_with(text) {
        results.push(Pair { display: "buckets/".into(), replacement: "buckets/".into() });
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_cmd_basic() {
        assert_eq!(split_cmd("ls"), ("ls", ""));
        assert_eq!(split_cmd("ls foo"), ("ls", "foo"));
        assert_eq!(split_cmd("  ls   foo bar"), ("", "ls   foo bar"));
        assert_eq!(split_cmd("cat path/with/spaces in it"), ("cat", "path/with/spaces in it"));
    }

    #[test]
    fn has_glob_chars_detects_common_metachars() {
        assert!(has_glob_chars("*.txt"));
        assert!(has_glob_chars("foo?"));
        assert!(has_glob_chars("file[0-9].bin"));
        assert!(has_glob_chars("dir/*.md"));
    }

    #[test]
    fn has_glob_chars_ignores_plain_paths() {
        assert!(!has_glob_chars("foo"));
        assert!(!has_glob_chars("a/b/c.txt"));
        assert!(!has_glob_chars(""));
    }

    #[test]
    fn basename_returns_last_component() {
        assert_eq!(basename("foo/bar/baz.txt"), "baz.txt");
        assert_eq!(basename("standalone"), "standalone");
        assert_eq!(basename(""), "");
    }

    #[test]
    fn normalize_path_drops_dot_and_pops_dotdot() {
        assert_eq!(normalize_path(""), "");
        assert_eq!(normalize_path("a/b"), "a/b");
        assert_eq!(normalize_path("a/./b"), "a/b");
        assert_eq!(normalize_path("a/../b"), "b");
        assert_eq!(normalize_path("a/b/.."), "a");
        assert_eq!(normalize_path("a//b"), "a/b");
    }

    #[test]
    fn normalize_path_clamps_at_root() {
        // ".." past the root just sticks at root.
        assert_eq!(normalize_path(".."), "");
        assert_eq!(normalize_path("../x"), "x");
        assert_eq!(normalize_path("a/../../b"), "b");
    }

    #[test]
    fn glob_pattern_matches_leaf() {
        let p = glob::Pattern::new("checkpoint-*").unwrap();
        assert!(p.matches("checkpoint-500"));
        assert!(!p.matches("model.bin"));
        let q = glob::Pattern::new("*.parquet").unwrap();
        assert!(q.matches("train-00000-of-00001.parquet"));
        assert!(!q.matches("README.md"));
    }

    #[test]
    fn guess_content_type_known_extensions() {
        assert_eq!(guess_content_type("a/b/x.json").as_deref(), Some("application/json"));
        assert_eq!(guess_content_type("README.md").as_deref(), Some("text/plain"));
        assert_eq!(guess_content_type("pic.JPG").as_deref(), Some("image/jpeg"));
        assert_eq!(
            guess_content_type("weights.safetensors").as_deref(),
            Some("application/octet-stream")
        );
    }

    #[test]
    fn guess_content_type_unknown_returns_none() {
        assert_eq!(guess_content_type("weights"), None);
        assert_eq!(guess_content_type("archive.xyz"), None);
    }

    #[test]
    fn expand_tilde_leaves_non_tilde_borrowed() {
        let s = "foo/bar";
        assert_eq!(expand_tilde(s).as_ref(), "foo/bar");
        assert!(matches!(expand_tilde(s), std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn expand_tilde_rewrites_prefix_only() {
        // We can only exercise this if $HOME is set; on CI runners it always is.
        if dirs::home_dir().is_none() {
            return;
        }
        let home = dirs::home_dir().unwrap().to_string_lossy().into_owned();
        let home = home.trim_end_matches('/').to_string();
        assert_eq!(expand_tilde("~").as_ref(), home);
        assert_eq!(expand_tilde("~/x/y").as_ref(), format!("{}/x/y", home));
        // No trailing-tilde expansion (unlike shells, we don't touch mid-path tildes).
        assert_eq!(expand_tilde("./~/y").as_ref(), "./~/y");
        assert_eq!(expand_tilde("~user/x").as_ref(), "~user/x");
    }

    #[test]
    fn parse_hf_url_bucket_needs_ns_and_name() {
        let u = parse_hf_url("hf://buckets/alice/my-bucket/path/to/file").unwrap().unwrap();
        assert_eq!(u.kind, RepoKind::Bucket);
        assert_eq!(u.repo_id, "alice/my-bucket");
        assert_eq!(u.path, "path/to/file");

        // Path can be empty — for use in directory-style listing/completion.
        let u = parse_hf_url("hf://buckets/alice/b").unwrap().unwrap();
        assert_eq!(u.repo_id, "alice/b");
        assert_eq!(u.path, "");

        // Missing name → error.
        assert!(parse_hf_url("hf://buckets/alice").is_err());
        assert!(parse_hf_url("hf://buckets//").is_err());
    }

    #[test]
    fn parse_hf_url_dataset_legacy_single_segment_id() {
        let u = parse_hf_url("hf://datasets/squad/train.parquet").unwrap().unwrap();
        assert_eq!(u.kind, RepoKind::Dataset);
        assert_eq!(u.repo_id, "squad");
        assert_eq!(u.path, "train.parquet");
    }

    #[test]
    fn parse_hf_url_dataset_modern_two_segment_id_with_nested_path() {
        let u = parse_hf_url("hf://datasets/HuggingFaceH4/zephyr-7b/data/train.parquet")
            .unwrap()
            .unwrap();
        assert_eq!(u.kind, RepoKind::Dataset);
        assert_eq!(u.repo_id, "HuggingFaceH4/zephyr-7b");
        assert_eq!(u.path, "data/train.parquet");
    }

    #[test]
    fn parse_hf_url_model_two_segment() {
        let u = parse_hf_url("hf://models/meta-llama/Llama-3.1-8B/config.json")
            .unwrap()
            .unwrap();
        assert_eq!(u.kind, RepoKind::Model);
        assert_eq!(u.repo_id, "meta-llama/Llama-3.1-8B");
        assert_eq!(u.path, "config.json");
    }

    #[test]
    fn parse_hf_url_rejects_unknown_scheme_and_non_hf_text() {
        assert!(parse_hf_url("s3://bucket/x").unwrap().is_none()); // no hf:// → returns None
        assert!(parse_hf_url("relative/path").unwrap().is_none());
        // hf:// with unknown kind → error
        assert!(parse_hf_url("hf://spaces/x/y").is_err());
    }

    #[test]
    fn walk_local_dir_yields_posix_relative_paths() {
        let tmp = std::env::temp_dir().join(format!("hfsh-walk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("a/b")).unwrap();
        std::fs::write(tmp.join("top.txt"), "x").unwrap();
        std::fs::write(tmp.join("a/mid.bin"), "x").unwrap();
        std::fs::write(tmp.join("a/b/leaf.log"), "x").unwrap();

        let mut found: Vec<String> = Vec::new();
        walk_local_dir(&tmp, &mut |_p, rel| found.push(rel.to_string())).unwrap();
        found.sort();
        assert_eq!(found, vec!["a/b/leaf.log", "a/mid.bin", "top.txt"]);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
