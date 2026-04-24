# hf-bsh

An interactive **bucket shell** for the Hugging Face Hub, written in Rust.
Distributed as an extension for the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
— install once, then launch with `hf bsh`. Also works as a standalone binary.

```
$ hf bsh buckets/alice/models
hf:alice/models> ls
            checkpoints/
   42 MB  2025-01-15 18:02  config.json
  4.9 GB  2025-01-15 18:02  weights.safetensors
hf:alice/models> cp hf://datasets/HuggingFaceH4/ultrachat_200k/data/train-00000-of-00003.parquet  raw/
copied hf://datasets/HuggingFaceH4/ultrachat_200k/data/train-00000-of-00003.parquet -> raw/train-00000-of-00003.parquet
hf:alice/models> put ./new-weights.safetensors .
uploading 1 file  100.0%  1.2 GB /  1.2 GB    840 MB/s
uploaded ./new-weights.safetensors -> new-weights.safetensors (1234567890 bytes)
```

## Why this exists

`hf buckets` (the built-in subcommand) covers one-shot `ls` / `cat` / `rm` etc,
but every invocation needs a fully-qualified `hf://buckets/ns/name/...` path
and there's no "current directory" concept. `hf-bsh` is a persistent REPL
around the same API:

- **`cd`, `ls`, `tree`, `pwd`** — explore the bucket like a filesystem.
- **`put` / `get`** — local⇄bucket transfer with a progress bar.
- **`cp` / `mv`** — server-side xet-ref copies within the bucket, or
  **`cp hf://datasets/<id>/<path> <dst>`** to pull from any dataset/model
  directly into the bucket without ever downloading locally.
- **Tab completion** for bucket paths, local paths, and hf:// source URLs.
- **Glob support** (`*`, `?`, `[..]`) on final path components.

For browsing or downloading datasets/models themselves, use `hf download` —
it's well-optimised for that and we don't try to compete.

## Install

### As an `hf` extension (recommended)

```
hf extensions install torrid-fish/hf-bsh
hf bsh buckets/<ns>/<name>
```

Or for official-org installs once adopted:

```
hf bsh buckets/<ns>/<name>     # auto-installs if missing
```

### Standalone

```
cargo install --path .                 # → ~/.cargo/bin/hf-bsh
# or
cargo build --release                  # → target/release/hf-bsh
```

For a fully-static musl build:

```
sudo apt-get install musl-tools              # Debian/Ubuntu
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

## Authentication

Picks up a Hugging Face token from the first of:

1. `--token <TOKEN>` CLI flag
2. `$HF_TOKEN` / `$HUGGING_FACE_HUB_TOKEN`
3. File at `$HF_TOKEN_PATH`, `$HF_HOME/token`, or `~/.cache/huggingface/token`
   (the standard `hf auth login` / `huggingface-cli login` locations)

Buckets always require a token; public dataset/model files pulled via
`cp hf://...` do not.

## Usage

```
hf-bsh [options] [buckets/<ns>/<name>]
hf bsh [options] [buckets/<ns>/<name>]
```

Passing a target on entry is equivalent to running `open <target>`
immediately. The target must be a bucket — dataset/model support is only
available as a `cp` source, not as an open target.

| Flag | Description |
|---|---|
| `--endpoint <URL>` | override the Hub endpoint (or set `$HF_ENDPOINT`) |
| `--token <TOKEN>` | override the auth token |
| `-h`, `--help` | show help |
| `-V`, `--version` | show version |

## Commands

| Command | Description |
|---|---|
| `open buckets/<ns>/<name>` | open a bucket |
| `cd <path>` \| `cd ..` \| `cd /` | change directory (handles `.` / `..` / absolute paths) |
| `ls [path]` | list entries (size, mtime, name) |
| `pwd` | print `hf://` URL of cwd |
| `cat <path>` | dump a text file (≤1 MiB, binaries refused) |
| `du [-h] [path]` | total bytes; `-h` → KB/MB/GB/TB |
| `find [path]` | recursive path dump |
| `tree [-L N] [path]` | tree view |
| `rm [-r] <path>…` | delete file(s) |
| `mv <src>… <dst>` | move files/dirs within the bucket |
| `cp <src>… <dst>` | copy files/dirs; `<src>` may be an `hf://…` URL for cross-repo server-side copy |
| `put <local-src>… <dst>` | upload local files/dirs into the bucket |
| `get <remote-src>… [<dst>]` | download remote files/dirs to local fs (default dst: `.`) |
| `refresh` | clear the completion cache |
| `help` \| `?` | command summary |
| `exit` \| `quit` | leave the shell |

## Path resolution

| Form | Meaning |
|---|---|
| `foo/bar` | joins to current cwd |
| `./foo` | same as `foo` |
| `../foo` | parent, then `foo` |
| `/foo` | from the bucket root |
| `..` past root | clamps to root (no error) |
| `~`, `~/foo` | expands to `$HOME` / `$HOME/foo` (local paths only: `put`, `get`) |

## Cross-repo copy

`cp` accepts `hf://{buckets,datasets,models}/<id>/<path>` URLs as sources.
These run **entirely server-side** — the client just forwards the xet hash,
so even multi-GB copies complete in seconds regardless of your bandwidth.

```
cp hf://datasets/squad/train.parquet               raw/
cp hf://models/meta-llama/Llama-3.1-8B/config.json cfg/llama.json
cp hf://buckets/other-ns/other-bucket/archive/file .    # cross-bucket too
```

`mv` is **bucket-internal only** — we don't own the source-side delete
permission for foreign repos. Globs and directory expansion aren't supported
for `hf://` sources yet (concrete file paths only). Only xet-backed source
files are supported; non-xet legacy files will error out and you should
`hf download` + `put` them instead.

## Glob patterns

Paths accept POSIX-style globs in the **final component** (`*`, `?`, `[..]`):

```
rm -r checkpoint-*
mv data/*.parquet archive/          # dir dst takes multiple sources
ls *.json                           # list only matching entries
du 'images-[0-9]*'                  # quote to defer shell expansion
```

No match errors zsh-style (`hf-bsh: no match: <pattern>`). `cat` refuses
multi-match. `**` recursive globs and globs in non-final components aren't
supported. Globs do **not** apply to `hf://…` sources yet.

## Upload / download

```
# upload
put model.safetensors checkpoints/          # → checkpoints/model.safetensors
put data/ archive/                          # recursive; mirrors subtree
put *.parquet raw/                          # local glob expands
put ~/weights/                              # ~ expands to $HOME

# download
get weights.bin                             # → ./weights.bin
get checkpoints/ ./backup/                  # recursive
get train-*.parquet ./data/                 # remote glob expands
```

A TTY progress bar shows percentage, throughput, and ETA for both `put`
and `get`. In non-TTY mode (pipes, logs) it collapses to a single
one-line summary.

## Scope & limitations

- **Buckets only.** `open` accepts nothing else. Browsing datasets/models
  is `hf`'s job.
- **No `sync` / dry-run.** `put` / `get` are one-shot; destination files
  are overwritten.
- **Cross-repo `cp` requires xet-backed sources.** Legacy LFS-only files
  error out with a pointer to `hf download` + `put`.
- Only the `main` revision is addressable for external dataset/model sources
  — no branch, tag, or commit SHA selection yet.

## How it works

- Listings, search, and bucket mutations are direct REST calls to
  `https://huggingface.co/api/...` over `reqwest` + `rustls` (no
  `huggingface_hub` dependency).
- Bucket `cat` streams via the Hugging Face xet CAS protocol through the
  `hf-xet` crate (`XetDownloadStreamGroup`).
- Bucket `mv`/`cp` (own-bucket) and `cp hf://...` (cross-repo) are
  server-side only: the client fetches the source's xet hash (from
  `paths-info` for buckets or `HEAD /resolve/main/<path>` for
  datasets/models) and posts NDJSON `copyFile` operations to
  `/api/buckets/<id>/batch`. No data ever transits the client.
- Bucket `put` runs an `XetUploadCommit` against the bucket's xet CAS
  (authenticated with an `xet-write-token` JWT), then posts `addFile`
  NDJSON with the resulting hashes to the same `/batch` endpoint. Content
  deduplication happens automatically client↔server.
- Bucket `get` resolves xet hashes via `paths-info` and streams each file
  straight to disk via `XetFileDownloadGroup` (parallel chunks, shared
  CAS connection).
