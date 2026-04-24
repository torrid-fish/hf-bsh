use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use reqwest::blocking::{Client as HttpClient, RequestBuilder, Response};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT as HDR_UA};
use serde::Deserialize;

use crate::fmt::parse_datetime;
use crate::progress::ProgressHandle;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_REVISION: &str = "main";
const USER_AGENT: &str = concat!("hfsh/", env!("CARGO_PKG_VERSION"));

/// One entry in a listing — covers both repo (model/dataset) and bucket shapes.
#[derive(Debug, Clone)]
pub struct TreeEntry {
    pub path: String,
    pub is_dir: bool,
    pub size: Option<u64>,
    pub mtime: Option<DateTime<Utc>>,
}

/// `"model"`, `"dataset"`, or `"bucket"`. We only ever send the first two over the wire
/// through `list_repo_tree`; bucket uses a separate endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepoKind {
    Model,
    Dataset,
    Bucket,
}

/// Per-file info returned by `POST /api/buckets/{id}/paths-info`. The `xet_hash`
/// is what server-side copy operations reference; `path` and `size` are used by
/// bucket `cat` in Phase 3.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BucketPathInfo {
    pub path: String,
    pub xet_hash: String,
    pub size: Option<u64>,
}

/// A single copy operation for `bucket_batch`. Copying within the same bucket
/// uses `source_repo_type = "bucket"` and `source_repo_id = <bucket_id>`.
#[derive(Debug, Clone)]
pub struct BucketCopyOp {
    pub source_repo_type: String,
    pub source_repo_id: String,
    pub xet_hash: String,
    pub destination: String,
}

/// A single `addFile` operation for `bucket_batch`. `xet_hash` must already
/// reference CAS-ingested data (e.g. from `xet_upload_files`). `mtime_ms` is
/// the file modification time in milliseconds since the Unix epoch.
#[derive(Debug, Clone)]
pub struct BucketAddOp {
    pub destination: String,
    pub xet_hash: String,
    pub mtime_ms: i64,
    pub content_type: Option<String>,
}

/// Result of a single successful CAS upload performed by `xet_upload_files`.
/// The caller combines these with mtime / content-type to build `BucketAddOp`s.
#[derive(Debug, Clone)]
pub struct UploadedFile {
    pub remote_path: String,
    pub xet_hash: String,
    pub size: u64,
}

/// CAS JWT handshake response, as returned by `GET /api/buckets/{id}/xet-{op}-token`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CasJwt {
    pub cas_url: String,
    pub exp: u64,
    pub access_token: String,
}

impl RepoKind {
    pub fn repo_type_path(&self) -> &'static str {
        match self {
            RepoKind::Model => "models",
            RepoKind::Dataset => "datasets",
            RepoKind::Bucket => "buckets",
        }
    }
}

pub struct Client {
    http: HttpClient,
    pub endpoint: String,
    pub token: Option<String>,
}

impl Client {
    /// Build a client; explicit `endpoint` / `token` override env/file sources.
    pub fn with_overrides(endpoint: Option<String>, token: Option<String>) -> Self {
        let endpoint = endpoint
            .or_else(|| std::env::var("HF_ENDPOINT").ok())
            .map(|s| s.trim_end_matches('/').to_string())
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
        let token = token.or_else(load_token);
        let mut default_headers = HeaderMap::new();
        default_headers.insert(
            HDR_UA,
            HeaderValue::from_static(USER_AGENT),
        );
        if let Some(tok) = &token {
            if let Ok(v) = HeaderValue::from_str(&format!("Bearer {}", tok)) {
                default_headers.insert(AUTHORIZATION, v);
            }
        }
        let http = HttpClient::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(120))
            .default_headers(default_headers)
            .build()
            .expect("reqwest client build");
        Self { http, endpoint, token }
    }

    // ------------------------------------------------------------------
    // Listings
    // ------------------------------------------------------------------

    pub fn list_bucket_tree(
        &self,
        bucket_id: &str,
        prefix: Option<&str>,
        recursive: bool,
    ) -> Result<Vec<TreeEntry>> {
        let encoded = prefix
            .filter(|p| !p.is_empty())
            .map(|p| format!("/{}", urlencoding::encode(p)))
            .unwrap_or_default();
        let url = format!(
            "{}/api/buckets/{}/tree{}",
            self.endpoint, bucket_id, encoded
        );
        let params = &[("recursive", if recursive { "true" } else { "false" })];
        let raw = self.paginate_json(&url, params)?;
        let mut out = Vec::with_capacity(raw.len());
        for v in raw {
            out.push(parse_bucket_entry(&v)?);
        }
        Ok(out)
    }

    pub fn list_datasets(&self, search: Option<&str>, limit: u32) -> Result<Vec<String>> {
        self.list_repo_index("datasets", search, limit)
    }

    pub fn list_models(&self, search: Option<&str>, limit: u32) -> Result<Vec<String>> {
        self.list_repo_index("models", search, limit)
    }

    fn list_repo_index(&self, kind: &str, search: Option<&str>, limit: u32) -> Result<Vec<String>> {
        let url = format!("{}/api/{}", self.endpoint, kind);
        let limit_s = limit.to_string();
        let mut params: Vec<(&str, &str)> = vec![("limit", limit_s.as_str())];
        if let Some(s) = search.filter(|s| !s.is_empty()) {
            params.push(("search", s));
        }
        let raw = self.get_json_array(&url, &params)?;
        Ok(raw
            .into_iter()
            .filter_map(|v| v.get("id").and_then(|x| x.as_str()).map(|s| s.to_string()))
            .collect())
    }

    pub fn list_buckets(&self, namespace: Option<&str>) -> Result<Vec<String>> {
        let ns = match namespace.filter(|s| !s.is_empty()) {
            Some(n) => n.to_string(),
            None => self.whoami_name()?,
        };
        let url = format!("{}/api/buckets/{}", self.endpoint, ns);
        let raw = self.get_json_array(&url, &[])?;
        Ok(raw
            .into_iter()
            .filter_map(|v| v.get("id").and_then(|x| x.as_str()).map(|s| s.to_string()))
            .collect())
    }

    fn whoami_name(&self) -> Result<String> {
        if self.token.is_none() {
            bail!("not logged in: set HF_TOKEN or log in with `huggingface-cli login`");
        }
        let url = format!("{}/api/whoami-v2", self.endpoint);
        let resp = send_checked(self.http.get(&url))?;
        let v: serde_json::Value = resp.json().context("parse whoami json")?;
        v.get("name")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("whoami response missing 'name'"))
    }

    // ------------------------------------------------------------------
    // Bucket mutations: paths-info (lookup xet_hash) + batch (NDJSON)
    // ------------------------------------------------------------------

    /// Fetch per-file metadata (including xet_hash) for a list of bucket paths.
    /// Paths that don't exist are silently omitted by the server.
    pub fn bucket_paths_info(
        &self,
        bucket_id: &str,
        paths: &[String],
    ) -> Result<Vec<BucketPathInfo>> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/api/buckets/{}/paths-info", self.endpoint, bucket_id);
        let body = serde_json::json!({ "paths": paths });
        let resp = send_checked(self.http.post(&url).json(&body))?;
        let arr: Vec<serde_json::Value> = resp.json().context("parse paths-info json")?;
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            let path = v
                .get("path")
                .and_then(|x| x.as_str())
                .ok_or_else(|| anyhow!("paths-info: entry missing 'path'"))?
                .to_string();
            let xet_hash = v
                .get("xetHash")
                .and_then(|x| x.as_str())
                .ok_or_else(|| anyhow!("paths-info: entry missing 'xetHash'"))?
                .to_string();
            let size = v.get("size").and_then(|x| x.as_u64());
            out.push(BucketPathInfo { path, xet_hash, size });
        }
        Ok(out)
    }

    /// Execute a batch of add/copy/delete operations on a bucket. Server-side only —
    /// no xet data-plane traffic (CAS ingestion for `adds` must have happened first,
    /// e.g. via `xet_upload_files`). `copies` may reference files in the same bucket
    /// (`source_repo_type = "bucket"`, `source_repo_id = bucket_id`) or any other repo.
    pub fn bucket_batch(
        &self,
        bucket_id: &str,
        adds: &[BucketAddOp],
        copies: &[BucketCopyOp],
        deletes: &[String],
    ) -> Result<()> {
        if adds.is_empty() && copies.is_empty() && deletes.is_empty() {
            return Ok(());
        }
        let url = format!("{}/api/buckets/{}/batch", self.endpoint, bucket_id);
        let mut body = Vec::<u8>::new();
        for op in adds {
            let mut payload = serde_json::json!({
                "type": "addFile",
                "path": op.destination,
                "xetHash": op.xet_hash,
                "mtime": op.mtime_ms,
            });
            if let Some(ct) = &op.content_type {
                payload["contentType"] = serde_json::Value::String(ct.clone());
            }
            serde_json::to_writer(&mut body, &payload).context("ndjson add")?;
            body.push(b'\n');
        }
        for op in copies {
            let payload = serde_json::json!({
                "type": "copyFile",
                "path": op.destination,
                "xetHash": op.xet_hash,
                "sourceRepoType": op.source_repo_type,
                "sourceRepoId": op.source_repo_id,
            });
            serde_json::to_writer(&mut body, &payload).context("ndjson copy")?;
            body.push(b'\n');
        }
        for path in deletes {
            let payload = serde_json::json!({ "type": "deleteFile", "path": path });
            serde_json::to_writer(&mut body, &payload).context("ndjson delete")?;
            body.push(b'\n');
        }
        let resp = send_checked(
            self.http
                .post(&url)
                .header(CONTENT_TYPE, "application/x-ndjson")
                .body(body),
        )?;
        // Consume body to free the connection; success is any 2xx.
        let _ = resp.bytes();
        Ok(())
    }

    /// Exchange the HF bearer token for a CAS JWT used by the xet data plane.
    pub fn cas_read_token(&self, bucket_id: &str) -> Result<CasJwt> {
        if self.token.is_none() {
            bail!("bucket access requires an HF token (set HF_TOKEN or --token)");
        }
        let url = format!(
            "{}/api/buckets/{}/xet-read-token",
            self.endpoint, bucket_id
        );
        let resp = send_checked(self.http.get(&url))?;
        resp.json::<CasJwt>().context("parse xet-read-token json")
    }

    /// Exchange the HF bearer token for a CAS JWT used by the xet data plane
    /// with write scope — needed to upload new files into the bucket.
    pub fn cas_write_token(&self, bucket_id: &str) -> Result<CasJwt> {
        if self.token.is_none() {
            bail!("bucket upload requires an HF token (set HF_TOKEN or --token)");
        }
        let url = format!(
            "{}/api/buckets/{}/xet-write-token",
            self.endpoint, bucket_id
        );
        let resp = send_checked(self.http.get(&url))?;
        resp.json::<CasJwt>().context("parse xet-write-token json")
    }

    /// Download a bucket file into memory via the xet data plane. Caps at
    /// `max_bytes + 1` so callers can detect overflow (mirrors the repo path).
    pub fn download_bucket_file(
        &self,
        bucket_id: &str,
        path_info: &BucketPathInfo,
        max_bytes: u64,
    ) -> Result<Vec<u8>> {
        use xet::xet_session::{XetFileInfo, XetSessionBuilder};

        let jwt = self.cas_read_token(bucket_id)?;
        let session = XetSessionBuilder::new()
            .build()
            .map_err(|e| anyhow!("xet session build failed: {}", e))?;
        let group = session
            .new_download_stream_group()
            .map_err(|e| anyhow!("xet new_download_stream_group: {}", e))?
            .with_endpoint(jwt.cas_url)
            .with_token_info(jwt.access_token, jwt.exp)
            .build_blocking()
            .map_err(|e| anyhow!("xet stream group build: {}", e))?;

        let info = XetFileInfo {
            hash: path_info.xet_hash.clone(),
            file_size: path_info.size,
            sha256: None,
        };
        let mut stream = group
            .download_stream_blocking(info, None)
            .map_err(|e| anyhow!("xet download_stream: {}", e))?;

        let mut buf: Vec<u8> = Vec::new();
        let cap = max_bytes.saturating_add(1) as usize;
        loop {
            let next = stream
                .blocking_next()
                .map_err(|e| anyhow!("xet stream read: {}", e))?;
            let Some(chunk) = next else { break };
            if buf.len() + chunk.len() > cap {
                // Stop early — caller will refuse based on size.
                buf.extend_from_slice(&chunk[..cap.saturating_sub(buf.len())]);
                break;
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(buf)
    }

    // ------------------------------------------------------------------
    // (Dataset / model local downloads were removed — use `hf download`.)
    // ------------------------------------------------------------------

    /// Fetch the xet hash + size for a single file in a dataset or model repo.
    /// Issues `HEAD /{repo_or_datasets/repo}/resolve/main/<path>` and pulls the
    /// values from `X-Xet-Hash` and `X-Linked-Size`. Fails clearly when the
    /// file isn't xet-backed (e.g. very old LFS-only repos) — callers can
    /// surface that to the user so they know to fall back to `hf download`.
    pub fn repo_xet_info(
        &self,
        kind: RepoKind,
        repo_id: &str,
        path_in_repo: &str,
    ) -> Result<BucketPathInfo> {
        if matches!(kind, RepoKind::Bucket) {
            bail!("repo_xet_info: use bucket_paths_info for bucket sources");
        }
        let prefix = match kind {
            RepoKind::Dataset => "datasets/",
            RepoKind::Model => "",
            RepoKind::Bucket => unreachable!(),
        };
        let url = format!(
            "{}/{}{}/resolve/{}/{}",
            self.endpoint,
            prefix,
            repo_id,
            DEFAULT_REVISION,
            encode_path_keep_slashes(path_in_repo),
        );
        let resp = send_checked(self.http.head(&url))?;
        let headers = resp.headers();
        let xet_hash = headers
            .get("x-xet-hash")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                anyhow!(
                    "{}: not xet-backed (no X-Xet-Hash header); use `hf download` + `put` instead",
                    path_in_repo
                )
            })?;
        let size = headers
            .get("x-linked-size")
            .or_else(|| headers.get("content-length"))
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        Ok(BucketPathInfo {
            path: path_in_repo.to_string(),
            xet_hash,
            size,
        })
    }

    // ------------------------------------------------------------------
    // Bucket uploads (xet CAS + /batch addFile)
    // ------------------------------------------------------------------

    /// Upload local files to the bucket's xet CAS and return the resulting
    /// `(remote_path, xet_hash, size)` triples. Order is preserved — the
    /// i-th input maps to the i-th output. This step does **not** make the
    /// files visible in the bucket; call `bucket_batch` with the returned
    /// data wrapped as `BucketAddOp`s to commit them.
    ///
    /// If `progress` is supplied, a background thread polls
    /// `XetUploadCommit::progress()` every 100 ms and pushes
    /// `(total_bytes, total_bytes_completed)` into the handle.
    pub fn xet_upload_files(
        &self,
        bucket_id: &str,
        files: &[(std::path::PathBuf, String)],
        progress: Option<ProgressHandle>,
    ) -> Result<Vec<UploadedFile>> {
        use xet::xet_session::{Sha256Policy, XetSessionBuilder};

        if files.is_empty() {
            return Ok(Vec::new());
        }

        let jwt = self.cas_write_token(bucket_id)?;
        let session = XetSessionBuilder::new()
            .build()
            .map_err(|e| anyhow!("xet session build failed: {}", e))?;
        let commit = session
            .new_upload_commit()
            .map_err(|e| anyhow!("xet new_upload_commit: {}", e))?
            .with_endpoint(jwt.cas_url)
            .with_token_info(jwt.access_token, jwt.exp)
            .build_blocking()
            .map_err(|e| anyhow!("xet upload commit build: {}", e))?;

        let mut handles = Vec::with_capacity(files.len());
        for (local, _remote) in files {
            let h = commit
                .upload_from_path_blocking(local.clone(), Sha256Policy::Compute)
                .map_err(|e| anyhow!("xet upload_from_path {}: {}", local.display(), e))?;
            handles.push(h);
        }

        // Poll the commit's GroupProgressReport on a side thread.
        let stop = Arc::new(AtomicBool::new(false));
        let poll = progress.as_ref().map(|p| {
            let p = p.clone();
            let commit = commit.clone();
            let stop = Arc::clone(&stop);
            thread::spawn(move || loop {
                let r = commit.progress();
                p.set_total(r.total_bytes);
                p.set_completed(r.total_bytes_completed);
                if stop.load(Ordering::Acquire) {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            })
        });

        let commit_result = commit.commit_blocking();
        stop.store(true, Ordering::Release);
        if let Some(j) = poll {
            let _ = j.join();
        }
        // Final catch-up read in case the bar missed the last tick.
        if let Some(p) = &progress {
            let r = commit.progress();
            p.set_total(r.total_bytes);
            p.set_completed(r.total_bytes_completed);
        }
        let report = commit_result.map_err(|e| anyhow!("xet commit: {}", e))?;

        let mut out = Vec::with_capacity(files.len());
        for (handle, (local, remote)) in handles.into_iter().zip(files.iter()) {
            let meta = report
                .uploads
                .get(&handle.task_id())
                .ok_or_else(|| anyhow!("xet commit: missing upload metadata for {}", local.display()))?;
            let size = meta.xet_info.file_size.unwrap_or_else(|| {
                std::fs::metadata(local).map(|m| m.len()).unwrap_or(0)
            });
            out.push(UploadedFile {
                remote_path: remote.clone(),
                xet_hash: meta.xet_info.hash.clone(),
                size,
            });
        }
        Ok(out)
    }

    /// Download a list of bucket files directly to local paths. Parent
    /// directories are created as needed. Zero-byte files are touched
    /// without a CAS round-trip. Optional `progress` is driven from the
    /// `XetFileDownloadGroup`'s report.
    pub fn download_bucket_files_to_paths(
        &self,
        bucket_id: &str,
        files: &[(BucketPathInfo, std::path::PathBuf)],
        progress: Option<ProgressHandle>,
    ) -> Result<()> {
        use xet::xet_session::{XetFileInfo, XetSessionBuilder};

        if files.is_empty() {
            return Ok(());
        }

        // Create parent dirs up front; also short-circuit zero-size files
        // so we don't hand empty descriptors to the xet group (it rejects them).
        let mut to_download: Vec<(XetFileInfo, std::path::PathBuf)> = Vec::new();
        for (info, dst) in files {
            if let Some(parent) = dst.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create dir {}", parent.display()))?;
                }
            }
            if info.size == Some(0) {
                std::fs::File::create(dst)
                    .with_context(|| format!("touch {}", dst.display()))?;
                continue;
            }
            to_download.push((
                XetFileInfo {
                    hash: info.xet_hash.clone(),
                    file_size: info.size,
                    sha256: None,
                },
                dst.clone(),
            ));
        }
        if to_download.is_empty() {
            return Ok(());
        }

        let jwt = self.cas_read_token(bucket_id)?;
        let session = XetSessionBuilder::new()
            .build()
            .map_err(|e| anyhow!("xet session build failed: {}", e))?;
        let group = session
            .new_file_download_group()
            .map_err(|e| anyhow!("xet new_file_download_group: {}", e))?
            .with_endpoint(jwt.cas_url)
            .with_token_info(jwt.access_token, jwt.exp)
            .build_blocking()
            .map_err(|e| anyhow!("xet download group build: {}", e))?;

        for (info, dst) in to_download {
            group
                .download_file_to_path_blocking(info, dst.clone())
                .map_err(|e| anyhow!("xet enqueue {}: {}", dst.display(), e))?;
        }

        let stop = Arc::new(AtomicBool::new(false));
        let poll = progress.as_ref().map(|p| {
            let p = p.clone();
            let group = group.clone();
            let stop = Arc::clone(&stop);
            thread::spawn(move || loop {
                let r = group.progress();
                p.set_total(r.total_bytes);
                p.set_completed(r.total_bytes_completed);
                if stop.load(Ordering::Acquire) {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            })
        });

        // finish_blocking consumes the group, so the caller's poll clone is
        // how we still read progress after the fact.
        let observer = group.clone();
        let finish_result = group.finish_blocking();
        stop.store(true, Ordering::Release);
        if let Some(j) = poll {
            let _ = j.join();
        }
        if let Some(p) = &progress {
            let r = observer.progress();
            p.set_total(r.total_bytes);
            p.set_completed(r.total_bytes_completed);
        }
        finish_result.map_err(|e| anyhow!("xet download finish: {}", e))?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Internals: pagination + request helpers
    // ------------------------------------------------------------------

    fn paginate_json(
        &self,
        url: &str,
        params: &[(&str, &str)],
    ) -> Result<Vec<serde_json::Value>> {
        let mut out = Vec::new();
        let mut resp = send_checked(self.http.get(url).query(params))?;
        loop {
            let next = next_link_from_resp(&resp);
            let page: Vec<serde_json::Value> = resp.json().context("parse page json")?;
            out.extend(page);
            let Some(n) = next else { break };
            resp = send_checked(self.http.get(&n))?;
        }
        Ok(out)
    }

    fn get_json_array(
        &self,
        url: &str,
        params: &[(&str, &str)],
    ) -> Result<Vec<serde_json::Value>> {
        let resp = send_checked(self.http.get(url).query(params))?;
        let v: serde_json::Value = resp.json().context("parse json")?;
        match v {
            serde_json::Value::Array(a) => Ok(a),
            _ => bail!("expected json array"),
        }
    }
}

// ----------------------------------------------------------------------
// Helpers (free functions)
// ----------------------------------------------------------------------

/// Encode a path-in-repo for a URL, keeping `/` separators but escaping other reserved chars.
pub(crate) fn encode_path_keep_slashes(p: &str) -> String {
    p.split('/')
        .map(|seg| urlencoding::encode(seg).into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn next_link_from_resp(resp: &Response) -> Option<String> {
    let hdr = resp.headers().get(reqwest::header::LINK)?;
    parse_next_link(hdr.to_str().ok()?)
}

/// Send a prepared request, error on non-2xx status (with server body included).
fn send_checked(rb: RequestBuilder) -> Result<Response> {
    let resp = rb.send().map_err(map_reqwest_err)?;
    let status = resp.status();
    if !status.is_success() {
        let url = resp.url().to_string();
        let body = resp.text().unwrap_or_default();
        let first = body.lines().next().unwrap_or("").to_string();
        bail!("HTTP {} at {}: {}", status.as_u16(), url, first);
    }
    Ok(resp)
}

/// Parse a GitHub-style `Link:` header value and return the URL with `rel="next"`, if any.
pub(crate) fn parse_next_link(hdr: &str) -> Option<String> {
    for part in hdr.split(',') {
        let part = part.trim();
        // format: <url>; rel="next"
        if let Some(end) = part.find('>') {
            if !part.starts_with('<') {
                continue;
            }
            let url = &part[1..end];
            let rest = &part[end + 1..];
            if rest.split(';').any(|p| {
                let p = p.trim();
                p == "rel=\"next\"" || p == "rel=next"
            }) {
                return Some(url.to_string());
            }
        }
    }
    None
}

fn map_reqwest_err(e: reqwest::Error) -> anyhow::Error {
    if let Some(url) = e.url() {
        anyhow!("{}: {}", url, e)
    } else {
        anyhow!("{}", e)
    }
}

pub(crate) fn parse_bucket_entry(v: &serde_json::Value) -> Result<TreeEntry> {
    let kind = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
    let path = v
        .get("path")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow!("bucket entry missing 'path'"))?
        .to_string();
    let is_dir = kind == "directory";
    let size = v.get("size").and_then(|x| x.as_u64());
    // Buckets prefer `mtime` for files, but fall back to `uploadedAt` for folders.
    let mtime_s = v
        .get("mtime")
        .and_then(|x| x.as_str())
        .or_else(|| v.get("uploadedAt").and_then(|x| x.as_str()));
    let mtime = mtime_s.and_then(parse_datetime);
    Ok(TreeEntry {
        path,
        is_dir,
        size: if is_dir { None } else { size },
        mtime,
    })
}

fn load_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        let t = t.trim().to_string();
        if !t.is_empty() {
            return Some(t);
        }
    }
    if let Ok(t) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        let t = t.trim().to_string();
        if !t.is_empty() {
            return Some(t);
        }
    }
    let home = dirs::home_dir()?;
    // Honour HF_HOME and HF_TOKEN_PATH if set.
    let token_path = std::env::var("HF_TOKEN_PATH")
        .map(std::path::PathBuf::from)
        .ok()
        .or_else(|| std::env::var("HF_HOME").ok().map(|h| std::path::PathBuf::from(h).join("token")))
        .unwrap_or_else(|| {
            let base = std::env::var("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| home.join(".cache"));
            base.join("huggingface").join("token")
        });
    let s = std::fs::read_to_string(token_path).ok()?;
    let t = s.trim().to_string();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_next_link_basic() {
        let hdr = r#"<https://api.example.com/foo?page=2>; rel="next", <https://api.example.com/foo?page=10>; rel="last""#;
        assert_eq!(
            parse_next_link(hdr),
            Some("https://api.example.com/foo?page=2".to_string())
        );
    }

    #[test]
    fn parse_next_link_no_next() {
        let hdr = r#"<https://api.example.com/foo?page=1>; rel="prev""#;
        assert_eq!(parse_next_link(hdr), None);
    }

    #[test]
    fn parse_next_link_unquoted_rel() {
        let hdr = r#"<https://x/p2>; rel=next"#;
        assert_eq!(parse_next_link(hdr), Some("https://x/p2".to_string()));
    }

    #[test]
    fn encode_path_keeps_slashes_but_escapes_spaces() {
        assert_eq!(encode_path_keep_slashes("a/b/c"), "a/b/c");
        assert_eq!(
            encode_path_keep_slashes("dir with space/file.txt"),
            "dir%20with%20space/file.txt"
        );
    }

    #[test]
    fn parse_bucket_entry_file_with_mtime() {
        let v = json!({
            "type": "file",
            "path": "models/weights.bin",
            "size": 42,
            "xetHash": "deadbeef",
            "mtime": "2025-01-02T03:04:05Z"
        });
        let e = parse_bucket_entry(&v).unwrap();
        assert!(!e.is_dir);
        assert_eq!(e.path, "models/weights.bin");
        assert_eq!(e.size, Some(42));
        assert!(e.mtime.is_some());
    }

    #[test]
    fn parse_bucket_entry_folder_falls_back_to_uploaded_at() {
        let v = json!({
            "type": "directory",
            "path": "models",
            "uploadedAt": "2024-11-11T00:00:00Z"
        });
        let e = parse_bucket_entry(&v).unwrap();
        assert!(e.is_dir);
        assert!(e.mtime.is_some());
    }
}
