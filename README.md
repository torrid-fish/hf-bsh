# hf-bsh

An interactive **bucket shell** for the Hugging Face Hub, written in Python on
top of [`huggingface_hub`](https://github.com/huggingface/huggingface_hub).

Distributed as an extension for the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
— install once, then launch with `hf bsh`. Also works as a standalone command.

```
$ hf bsh alice/models
hf:alice/models> ls
                  checkpoints/
  2025-01-15 18:02     42 MB  config.json
  2025-01-15 18:02    4.9 GB  weights.safetensors
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
hf bsh <ns>/<name>
```

Or for official-org installs once adopted:

```
hf bsh <ns>/<name>             # auto-installs if missing
```

The extension installs into an isolated virtualenv under
`~/.local/share/hf/extensions/hf-bsh/`, so it never touches your global
Python environment.

### Standalone

```
pip install hf-bsh           # from PyPI (or: pip install git+https://github.com/torrid-fish/hf-bsh)
hf-bsh <ns>/<name>
```

Or run from a checkout without installing:

```
pip install -e .
python -m hf_bsh <ns>/<name>
```

Requires Python ≥ 3.10 (the Hugging Face bucket API lives in
`huggingface_hub` 1.x, which dropped older Pythons). The xet data plane is provided by the `hf_xet`
package, which ships prebuilt wheels for Linux / macOS / Windows — pip picks
the right one automatically, so there's no per-platform binary to build.

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
hf-bsh [options] [<ns>/<name>]
hf bsh [options] [<ns>/<name>]
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
| `open <ns>/<name>` | open a bucket (also accepts the legacy `buckets/<ns>/<name>` form) |
| `cd <path>` \| `cd ..` \| `cd /` | change directory (handles `.` / `..` / absolute paths) |
| `ls [path]` | list entries (mtime, size, name) |
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

Transfer progress is shown by `huggingface_hub`/`hf_xet` (a `tqdm` bar in a
TTY); `hf-bsh` prints a per-file summary line when each transfer completes.

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

`hf-bsh` is a thin REPL layer over `huggingface_hub`'s bucket API — all the
networking and the xet CAS data plane live in the library:

- **Listings / `cat` / `du` / `find` / `tree`** call `list_bucket_tree` and
  `get_bucket_paths_info`.
- **`mv` / `cp` (own-bucket)** resolve each source's xet hash via
  `get_bucket_paths_info`, then issue a single `batch_bucket_files(copy=…,
  delete=…)` — server-side only, no data transits the client.
- **`cp hf://datasets|models|buckets/…`** (cross-repo) uses `copy_files`,
  which resolves the source hash and performs the copy entirely server-side.
- **`put`** hands local paths to `batch_bucket_files(add=…)`; the library
  ingests them into the bucket's xet CAS (with automatic content
  deduplication) and commits the `addFile` records.
- **`get`** streams files straight to disk via `download_bucket_files`
  (parallel xet chunks, shared CAS connection).

Auth tokens are resolved by `huggingface_hub` — `$HF_TOKEN` /
`$HUGGING_FACE_HUB_TOKEN`, then the saved token file from `hf auth login`.
