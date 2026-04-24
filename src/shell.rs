use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;

use anyhow::{anyhow, bail, Result};
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Helper};

use crate::api::{BucketCopyOp, Client, RepoKind, TreeEntry};
use crate::fmt::{fmt_entry, fmt_size};

pub const CAT_MAX_SIZE: u64 = 1 << 20; // 1 MiB

/// Mutable shell state. Held inside an `Rc<RefCell<_>>` so the rustyline helper
/// and the REPL loop can both touch it.
pub struct State {
    pub client: Client,
    pub mode: Option<RepoKind>,
    pub repo_or_bucket: String,
    pub cwd: String,
    pub ls_cache: HashMap<String, Vec<String>>,
}

impl State {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            mode: None,
            repo_or_bucket: String::new(),
            cwd: String::new(),
            ls_cache: HashMap::new(),
        }
    }

    fn is_opened(&self) -> bool {
        self.mode.is_some()
    }

    fn ensure_bucket_mode(&self, op: &str) -> Result<()> {
        match self.mode {
            Some(RepoKind::Bucket) => Ok(()),
            Some(_) => bail!("{}: not supported for read-only mode", op),
            None => bail!("{}: nothing opened", op),
        }
    }

    fn label(&self) -> String {
        match self.mode {
            Some(RepoKind::Bucket) => self.repo_or_bucket.clone(),
            Some(RepoKind::Dataset) => format!("ds:{}", self.repo_or_bucket),
            Some(RepoKind::Model) => format!("m:{}", self.repo_or_bucket),
            None => String::new(),
        }
    }

    fn url_root(&self) -> String {
        match self.mode {
            Some(RepoKind::Bucket) => format!("hf://buckets/{}", self.repo_or_bucket),
            Some(RepoKind::Dataset) => format!("hf://datasets/{}", self.repo_or_bucket),
            Some(RepoKind::Model) => format!("hf://models/{}", self.repo_or_bucket),
            None => "hf://".to_string(),
        }
    }

    pub fn prompt(&self) -> String {
        if !self.is_opened() {
            return "hf> ".to_string();
        }
        let shown = if self.cwd.is_empty() { String::new() } else { format!("/{}", self.cwd) };
        format!("hf:{}{}> ", self.label(), shown)
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
        match self.mode {
            Some(RepoKind::Bucket) => {
                let prefix = if rel_dir.is_empty() { None } else { Some(rel_dir) };
                let prefix_owned = prefix.map(|p| format!("{}/", p.trim_end_matches('/')));
                self.client.list_bucket_tree(
                    &self.repo_or_bucket,
                    prefix_owned.as_deref(),
                    recursive,
                )
            }
            Some(kind @ (RepoKind::Dataset | RepoKind::Model)) => {
                let rel = rel_dir.trim_end_matches('/');
                let path_in_repo = if rel.is_empty() { None } else { Some(rel) };
                self.client
                    .list_repo_tree(kind, &self.repo_or_bucket, path_in_repo, recursive)
            }
            None => bail!("nothing opened"),
        }
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
            bail!("usage: open {{buckets|datasets|models}}/<ns>/<name>");
        }
        let (kind, rest) = if let Some(r) = arg.strip_prefix("buckets/") {
            (RepoKind::Bucket, r)
        } else if let Some(r) = arg.strip_prefix("datasets/") {
            (RepoKind::Dataset, r)
        } else if let Some(r) = arg.strip_prefix("models/") {
            (RepoKind::Model, r)
        } else {
            bail!(
                "open: target must start with buckets/, datasets/, or models/ (got {:?})",
                arg
            );
        };
        if rest.is_empty() {
            bail!("open: missing repo/bucket id after prefix (got {:?})", arg);
        }
        // Buckets are always namespaced; datasets/models may be root-level
        // (e.g. `squad`, `bert-base-uncased`).
        if matches!(kind, RepoKind::Bucket) && !rest.contains('/') {
            bail!("open: buckets require <ns>/<name> (got {:?})", arg);
        }
        let mut st = self.state.borrow_mut();
        st.mode = Some(kind);
        st.repo_or_bucket = rest.to_string();
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
        let kind = st.mode.ok_or_else(|| anyhow!("no mode"))?;
        // Single-match cat only — batch concatenation with a 1 MiB cap is a
        // foot-gun (see CLAUDE.md).
        let data = if matches!(kind, RepoKind::Bucket) {
            let info = st
                .client
                .bucket_paths_info(&st.repo_or_bucket, &[rel.clone()])?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("cat: {}: not found", rel))?;
            st.client
                .download_bucket_file(&st.repo_or_bucket, &info, CAT_MAX_SIZE)?
        } else {
            st.client
                .download_repo_file(kind, &st.repo_or_bucket, &rel, CAT_MAX_SIZE)?
        };
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
            st.ensure_bucket_mode("rm")?;
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
        let bucket = self.state.borrow().repo_or_bucket.clone();
        self.state
            .borrow()
            .client
            .bucket_batch(&bucket, &[], &to_delete)?;
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
            st.ensure_bucket_mode(op)?;
        }

        // Resolve sources, keeping is_dir info so we can expand dirs later.
        let bucket = self.state.borrow().repo_or_bucket.clone();
        let mut src_entries: Vec<(String, bool)> = Vec::new();
        {
            let st = self.state.borrow();
            for s in srcs_args {
                if has_glob_chars(s) {
                    for e in st.glob_match(s)? {
                        src_entries.push((e.path, e.is_dir));
                    }
                } else {
                    let rel = st.remote_prefix(s);
                    if rel.is_empty() {
                        bail!("{}: cannot use root as a source", op);
                    }
                    match st.stat(&rel)? {
                        Some((is_dir, _)) => src_entries.push((rel, is_dir)),
                        None => bail!("{}: source not found: {}", op, rel),
                    }
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

        // Expand each source (file or directory) to concrete (src_file, dst_file)
        // pairs. For a directory we walk its files and mirror the subtree under
        // the landing base.
        let mut pairs: Vec<(String, String)> = Vec::new();
        for (src_path, is_dir) in &src_entries {
            let landing = if dst_is_dir {
                let leaf = basename(src_path);
                if dst_base.is_empty() {
                    leaf.to_string()
                } else {
                    format!("{}/{}", dst_base, leaf)
                }
            } else {
                dst_base.clone()
            };
            if *is_dir {
                let prefix = format!("{}/", src_path);
                let st = self.state.borrow();
                let mut saw_file = false;
                for e in st.iter_tree(src_path, true)? {
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
                    pairs.push((e.path.clone(), dst));
                }
                if !saw_file {
                    bail!("{}: directory {:?} is empty", op, src_path);
                }
            } else {
                pairs.push((src_path.clone(), landing));
            }
        }

        // Fetch xet_hash for every (file) source in one paths-info call.
        let src_paths: Vec<String> = pairs.iter().map(|(s, _)| s.clone()).collect();
        let mut hashes: std::collections::HashMap<String, String> = {
            let st = self.state.borrow();
            st.client
                .bucket_paths_info(&bucket, &src_paths)?
                .into_iter()
                .map(|i| (i.path, i.xet_hash))
                .collect()
        };
        let mut copies: Vec<BucketCopyOp> = Vec::with_capacity(pairs.len());
        for (src, dst) in &pairs {
            let xet_hash = hashes
                .remove(src)
                .ok_or_else(|| anyhow!("{}: source not found: {}", op, src))?;
            copies.push(BucketCopyOp {
                source_repo_type: "bucket".into(),
                source_repo_id: bucket.clone(),
                xet_hash,
                destination: dst.clone(),
            });
        }

        let deletes: Vec<String> = if delete_sources {
            pairs.iter().map(|(s, _)| s.clone()).collect()
        } else {
            Vec::new()
        };
        self.state
            .borrow()
            .client
            .bucket_batch(&bucket, &copies, &deletes)?;
        let verb = if delete_sources { "moved" } else { "copied" };
        for (src, dst) in &pairs {
            println!("{} {} -> {}", verb, src, dst);
        }
        self.state.borrow_mut().invalidate_cache();
        Ok(())
    }
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
    println!("  open buckets/<ns>/<name>     bucket (read/write)");
    println!("  open datasets/<repo>         dataset (read-only)");
    println!("  open models/<repo>           model (read-only)");
    println!("  cd <path> | cd .. | cd /     change dir (. / .. / / / absolute paths OK)");
    println!("  ls [path]                    list");
    println!("  pwd                          print hf:// URL");
    println!("  cat <path>                   dump a text file (<=1 MiB)");
    println!("  du [-h] [path]               total bytes (-h: human-readable)");
    println!("  find [path]                  recursive path dump");
    println!("  tree [-L N] [path]           tree view");
    println!("  rm [-r] <path>…              delete (bucket only)");
    println!("  mv <src>... <dst>            move files/dirs (bucket only; dst must be dir for multi-src)");
    println!("  cp <src>... <dst>            copy files/dirs (bucket only; same rules as mv)");
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
    "open", "cd", "pwd", "ls", "cat", "du", "find", "tree", "rm", "mv", "cp", "refresh", "help",
    "exit", "quit",
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
            "ls" | "cat" | "du" | "find" | "tree" | "rm" | "mv" | "cp" => {
                complete_remote_path(&self.state, token)
            }
            _ => Vec::new(),
        };
        Ok((start, candidates))
    }
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

fn complete_open(state: &Rc<RefCell<State>>, text: &str) -> Vec<Pair> {
    let client = &state.borrow().client;
    let mut results: Vec<Pair> = Vec::new();
    if let Some(q) = text.strip_prefix("datasets/") {
        if let Ok(ids) = client.list_datasets(if q.is_empty() { None } else { Some(q) }, 30) {
            for id in ids {
                if id.starts_with(q) {
                    let s = format!("datasets/{}", id);
                    results.push(Pair { display: s.clone(), replacement: s });
                }
            }
        }
        return results;
    }
    if let Some(q) = text.strip_prefix("models/") {
        if let Ok(ids) = client.list_models(if q.is_empty() { None } else { Some(q) }, 30) {
            for id in ids {
                if id.starts_with(q) {
                    let s = format!("models/{}", id);
                    results.push(Pair { display: s.clone(), replacement: s });
                }
            }
        }
        return results;
    }
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
    for hint in &["buckets/", "datasets/", "models/"] {
        if hint.starts_with(text) {
            results.push(Pair { display: (*hint).to_string(), replacement: (*hint).to_string() });
        }
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
}
