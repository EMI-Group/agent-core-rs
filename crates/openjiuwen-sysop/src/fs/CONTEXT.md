# fs — filesystem backend

## Intent

Implements the filesystem half of `openjiuwen-sysop`: the local backend behind
`fs::local_fs`, ported from Python `openjiuwen/core/sys_operation/local/fs_operation.py`.
The parent `../CONTEXT.md` "Behavior contract" section is the authoritative per-method parity
checklist; this file records only what is specific to this subtree.

## API Surface

- `pub fn local_fs(config: LocalWorkConfig) -> Result<Arc<dyn FsOperation>>` (`mod.rs`) — validates the
  config, builds `LocalFsOperation`, and is the only entry point `SysOperation` uses.
- `local::LocalFsOperation` — `new(config)`, `policy()`, `config()`, `impl FsOperation` (all ten
  Python methods, `supports_conditional_write() == true`) and `impl FsExtOperation`
  (`stat`, `exists`, `mkdir`, `delete`, `copy`, `move_path`, `search_files_with`).
- `list::{list_files, list_directories, file_system_item}` — directory listing with recursion,
  depth, suffix filter and sorting; `file_system_item` is the single `FileSystemItem` constructor
  shared with search.
- `search::search_files(policy, params, mode)` — Python `rglob` glob matching (`Glob`) plus the
  additive `Regex` variant; returns files only.
- `ext::{stat, exists, mkdir, delete, copy, move_path}` — Rust-only extension operations.
- `slice::{LineSelection, validate_read_params, split_lines_keep_ends, select_lines,
  take_first_chunk, effective_chunk_size, encoded_len, write_bytes}` — pure, allocation-light
  helpers for read validation/slicing and write-content composition.

## Constraints

- Every caller-supplied path goes through `PathPolicy`; the only exceptions are the HOST side of a
  transfer (upload source, download destination), which Python never sandbox-checks and which are
  only tilde-expanded and canonicalized.
- Failures are `OpResult` values (`fs_failure(execution, reason)`, code 199003; code 199011 with
  message `file version conflict` for a rejected conditional write). Nothing panics.
- Blocking work stays off the async task: `walkdir` walks run in `spawn_blocking`; file I/O is
  `tokio::fs`; only cheap `set_permissions`/metadata calls on already-resolved paths are sync.
- Locks are per-resolved-path `tokio::sync::RwLock`s in a `std::sync::Mutex<HashMap<..>>`; the std
  guard is never held across an `await`, and a poisoned mutex is recovered, never unwrapped.
  Readers take the shared lock (skipped when `only_read`), writers the exclusive one, and transfers
  lock both paths sorted by path (`Write` wins) so transfers cannot deadlock.
- Module docs explain the local backend; public items carry `# Errors`; clippy `all` + `pedantic`
  runs with `-D warnings`.

## Behavior notes specific to this subtree

- Validation messages: bytes-mode slicing → `Parameters 'head', 'tail', and 'line_range' are only
  supported in text mode`; mutual exclusion → `tail and head …`, `tail and line_range …`,
  `head and line_range … cannot be specified simultaneously`; conditional-write flag clash →
  `conditional write options are incompatible`; non-directory roots → `Path is not a directory: …`.
- Missing read target → `File does not exist: <path>`; directory read → `Is a directory: <path>`;
  read over `max_file_size_bytes` → a reason naming both the size and the limit.
- `read_file` reports `ReadFileData.encoding` only for `encoding = "auto"`.
- Transfer streams emit a terminal `TransferChunkData` (`chunk_size 0`) for an EMPTY source; text
  read streams emit ZERO items for an empty file, except tail mode which emits one empty terminal
  chunk (Python parity).
- `write_file` opens `create_new` for `create_if_absent`, `append + create` for `append`, otherwise
  `write + create + truncate`, and flushes explicitly (`tokio::fs::File` buffers writes).
- `move_path` falls back to copy-then-delete on any rename error whose kind is not
  `NotFound`/`PermissionDenied`/`AlreadyExists`, because stable `ErrorKind` has no cross-device
  variant.

## Known Issues / Deviations

- `local.rs` is ~1065 implementation lines plus a ~850-line test module: the assigned file set makes
  it the only home for the facade, lock map, read/write and the transfer/stream machinery (the
  rationale is in its module docs). Splitting it would need a new `read.rs`/`transfer.rs` module.
- `stat` on a LIVE symlink reports the resolved target (`is_symlink == false`) because `PathPolicy`
  canonicalizes the longest existing prefix; only a dangling link is reported as a symlink.
- `split_lines_keep_ends` does not split exotic Unicode line separators (`\v`, `\f`, `\u{1c}`–`\u{1e}`,
  `\u{85}`, `\u{2028}`, `\u{2029}`); `\n`, `\r\n` and lone `\r` are handled.
- `line_range: Some(0, 0)` counts as "not specified" (only non-zero values participate in
  validation), so it cannot conflict with `head`/`tail`.
- Lock-map entries are never evicted; memory grows with the number of distinct resolved paths a
  process touches. Locks are in-process only (Python uses cross-process SQLite locks).
- Directory copies skip symlinks found inside the tree (logged at `debug`) so a link cannot pull
  content from outside the sandbox.

## Test Strategy

- 45 module-level unit tests plus the slice helper tests, all offline and deterministic
  (`tempfile` workspaces, no network, no host locale): `slice.rs` (validation order, line splitting,
  selection, chunk math, write-content composition), `local.rs` (read text/bytes/slicing/max-size/
  binary policy/traversal denial/symlink escape, write newline + conditional semantics with a
  byte-identical file, transfers incl. permissions, stream shapes), `list.rs`, `search.rs` and
  `ext.rs` cover their own modules.
- Gates: `cargo fmt -p openjiuwen-sysop`, `cargo test -p openjiuwen-sysop --offline`,
  `cargo clippy -p openjiuwen-sysop --all-targets --offline -- -D warnings`,
  `cargo test -p openjiuwen-sysop --features sandbox --offline`.

## Routing Table

- `./mod.rs` → `local_fs` factory, module declarations
- `./types.rs` → frozen parameter/data types (`ReadFileParams`, `WriteFileParams`, `FileSystemItem`, ext types)
- `./local.rs` → `LocalFsOperation`, lock map, read/write/upload/download and their streams, trait impls
- `./slice.rs` → pure read validation/slicing, chunk sizes, write-content composition
- `./list.rs` → `list_files`, `list_directories`, directory walker, `file_system_item`, sorting
- `./search.rs` → `search_files` (glob + regex), exclude patterns
- `./ext.rs` → `stat`, `exists`, `mkdir`, `delete`, `copy`, `move_path`
