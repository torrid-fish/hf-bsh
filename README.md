# hfsh

An interactive shell (REPL) for the Hugging Face Hub, written in Rust.

You `open` a bucket / dataset / model and navigate it with POSIX-ish commands:
`cd`, `ls`, `cat`, `rm`, `cp`, `mv`, `tree`, `du`, `find`.

- **Buckets** — read/write (`rm`, `mv`, `cp`, `ls`, `tree`, `find`, `du`, `cat`)
- **Datasets** / **Models** — read-only (`ls`, `tree`, `find`, `du`, `cat`)

## Install

```
cargo install --path .                # copies to ~/.cargo/bin/hfsh
# or
cargo build --release                 # -> target/release/hfsh
```

A truly static binary (musl) — both `ring` and `aws-lc-sys` (via the xet
stack) need a musl C toolchain, so install `musl-tools` first:

```
sudo apt-get install musl-tools              # Debian/Ubuntu
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

The default glibc build only dynamically links `libc` / `libgcc` / `libm`, so
a static binary is rarely necessary in practice.

## Authentication

`hfsh` picks up a Hugging Face token from the first of:

1. `--token <TOKEN>` CLI flag
2. `$HF_TOKEN`
3. `$HUGGING_FACE_HUB_TOKEN`
4. File at `$HF_TOKEN_PATH`, `$HF_HOME/token`, or `~/.cache/huggingface/token`
   (the standard `huggingface-cli login` locations)

Public datasets and models work without a token. Buckets and private repos
require one.

## Usage

```
hfsh [options] [<target>]
```

Options:

| Flag | Description |
|---|---|
| `--endpoint <URL>` | override the Hub endpoint (or set `$HF_ENDPOINT`) |
| `--token <TOKEN>` | override the auth token |
| `-h`, `--help` | show help |
| `-V`, `--version` | show version |

Starting with a target is equivalent to running `open <target>` on entry.
Every target needs a `buckets/` / `datasets/` / `models/` prefix:

```
hfsh datasets/squad
hfsh models/bert-base-uncased
hfsh buckets/my-namespace/my-bucket
```

## Commands

| Command | Mode | Description |
|---|---|---|
| `open {buckets\|datasets\|models}/<ns>/<name>` | any | change current repo/bucket |
| `cd <path>` \| `cd ..` \| `cd /` | opened | change directory |
| `ls [path]` | opened | list entries (size, mtime, name) |
| `pwd` | opened | print `hf://` URL of cwd |
| `cat <path>` | any | dump a text file (≤1 MiB, binaries refused) |
| `du [-h] [path]` | opened | total bytes; `-h` prints KB/MB/GB/TB |
| `find [path]` | opened | recursive path dump |
| `tree [-L N] [path]` | opened | tree view |
| `rm [-r] <path>…` | bucket only | delete file(s) |
| `mv <src>… <dst>` | bucket only | move files or directories; `dst/` (or an existing dir) accepts multiple sources |
| `cp <src>… <dst>` | bucket only | copy files or directories; same rules as `mv` |
| `refresh` | any | clear the completion cache |
| `help` \| `?` | any | command summary |
| `exit` \| `quit` | any | leave the shell |

## Path resolution

Every command accepts the usual shell-style relative bits:

| Form | Meaning |
|---|---|
| `foo/bar` | joins to current cwd |
| `./foo` | same as `foo` |
| `../foo` | parent, then `foo` |
| `/foo` | from the bucket/repo root |
| `..` past root | clamps to root (no error) |

So `cat ../README.md`, `mv ../a.bin archive/`, and `du ../*` all work as you'd expect.

## Glob patterns

Paths accept POSIX-style globs in the **final component** (`*`, `?`, `[..]`):

```
rm -r checkpoint-*
mv data/*.parquet archive/          # dir dst takes multiple sources
ls *.json                           # list only matching entries
du 'images-[0-9]*'                  # quote to defer shell expansion
find '**/*.bin'                     # NOT supported — use cd + glob instead
```

No match errors zsh-style (`hfsh: no match: <pattern>`) rather than passing
the literal pattern through. `cat` refuses to concatenate multiple matches
(the 1 MiB cap makes it a foot-gun). `**` recursive globs and globs in
non-final components aren't supported.

## Scope & limitations

- Uploads (local → bucket) are not implemented. `cp`/`mv` move data that's
  already in the Hub (via xet hash references, server-side only). To upload
  new content from a local file, use the Python reference (`reference.py`).
- Only the `main` revision is addressed for datasets/models — no branch, tag,
  or commit SHA selection yet.

## How it works

- Listings, search, and bucket mutations are direct REST calls to
  `https://huggingface.co/api/...` over `reqwest` + `rustls` (no
  `huggingface_hub` dependency).
- Bucket `cat` uses the Hugging Face xet CAS protocol via the `hf-xet` crate
  (`XetDownloadStreamGroup` for in-memory streaming).
- Bucket `mv`/`cp` are server-side only: the client fetches the source file's
  xet hash with `paths-info` and posts NDJSON `copyFile` operations to
  `/api/buckets/<id>/batch`. No data ever transits the client.
- Repository (model/dataset) `cat` uses the plain `/{repo}/resolve/main/<path>`
  URL, which follows the CDN/LFS redirect chain automatically.