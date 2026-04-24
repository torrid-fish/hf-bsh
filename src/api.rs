use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use reqwest::blocking::{Client as HttpClient, RequestBuilder, Response};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT as HDR_UA};
use serde::Deserialize;

use crate::fmt::parse_datetime;

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

    pub fn list_repo_tree(
        &self,
        kind: RepoKind,
        repo_id: &str,
        path_in_repo: Option<&str>,
        recursive: bool,
    ) -> Result<Vec<TreeEntry>> {
        assert!(!matches!(kind, RepoKind::Bucket));
        let encoded = path_in_repo
            .filter(|p| !p.is_empty())
            .map(|p| format!("/{}", urlencoding::encode(p)))
            .unwrap_or_default();
        let url = format!(
            "{}/api/{}/{}/tree/{}{}",
            self.endpoint,
            kind.repo_type_path(),
            repo_id,
            DEFAULT_REVISION,
            encoded,
        );
        let params = &[
            ("recursive", if recursive { "true" } else { "false" }),
            ("expand", "true"),
        ];
        let raw = self.paginate_json(&url, params)?;
        let mut out = Vec::with_capacity(raw.len());
        for v in raw {
            out.push(parse_repo_entry(&v)?);
        }
        Ok(out)
    }

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

    /// Execute a batch of copy/delete operations on a bucket. Server-side only —
    /// no xet data-plane traffic. `copies` may reference files in the same bucket
    /// (`source_repo_type = "bucket"`, `source_repo_id = bucket_id`) or any other repo.
    pub fn bucket_batch(
        &self,
        bucket_id: &str,
        copies: &[BucketCopyOp],
        deletes: &[String],
    ) -> Result<()> {
        if copies.is_empty() && deletes.is_empty() {
            return Ok(());
        }
        let url = format!("{}/api/buckets/{}/batch", self.endpoint, bucket_id);
        let mut body = Vec::<u8>::new();
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
    // Downloads (repo only — buckets go through xet)
    // ------------------------------------------------------------------

    /// Download a single file as raw bytes. Refuses if the file exceeds `max_bytes`.
    pub fn download_repo_file(
        &self,
        kind: RepoKind,
        repo_id: &str,
        path_in_repo: &str,
        max_bytes: u64,
    ) -> Result<Vec<u8>> {
        let url = match kind {
            RepoKind::Model => format!(
                "{}/{}/resolve/{}/{}",
                self.endpoint,
                repo_id,
                DEFAULT_REVISION,
                encode_path_keep_slashes(path_in_repo),
            ),
            RepoKind::Dataset => format!(
                "{}/datasets/{}/resolve/{}/{}",
                self.endpoint,
                repo_id,
                DEFAULT_REVISION,
                encode_path_keep_slashes(path_in_repo),
            ),
            RepoKind::Bucket => {
                bail!("cat on buckets is not supported in this build (requires xet protocol)")
            }
        };
        let resp = send_checked(self.http.get(&url))?;
        // Cap at max_bytes + 1 so the caller can detect overflow.
        let limit = max_bytes.saturating_add(1);
        let bytes = read_limited(resp, limit)?;
        Ok(bytes)
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

/// Read at most `limit` bytes from `resp` into a `Vec<u8>`. Stops early if the
/// server is larger than `limit`.
fn read_limited(resp: Response, limit: u64) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut reader = resp.take(limit);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).context("read response body")?;
    Ok(buf)
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

#[derive(Deserialize)]
struct RepoEntryJson {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default, rename = "last_commit")]
    last_commit: Option<LastCommit>,
}

#[derive(Deserialize)]
struct LastCommit {
    date: Option<String>,
}

pub(crate) fn parse_repo_entry(v: &serde_json::Value) -> Result<TreeEntry> {
    let e: RepoEntryJson = serde_json::from_value(v.clone()).context("parse repo tree entry")?;
    let is_dir = e.kind == "directory";
    let mtime = e
        .last_commit
        .as_ref()
        .and_then(|lc| lc.date.as_deref())
        .and_then(parse_datetime);
    Ok(TreeEntry {
        path: e.path,
        is_dir,
        size: if is_dir { None } else { e.size },
        mtime,
    })
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
    fn parse_repo_entry_file_with_commit() {
        let v = json!({
            "type": "file",
            "path": "config.json",
            "size": 1234,
            "last_commit": { "date": "2024-06-01T12:34:56.000Z" }
        });
        let e = parse_repo_entry(&v).unwrap();
        assert!(!e.is_dir);
        assert_eq!(e.path, "config.json");
        assert_eq!(e.size, Some(1234));
        assert!(e.mtime.is_some());
    }

    #[test]
    fn parse_repo_entry_directory_has_no_size() {
        let v = json!({ "type": "directory", "path": "subdir" });
        let e = parse_repo_entry(&v).unwrap();
        assert!(e.is_dir);
        assert_eq!(e.size, None);
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
