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
- `read::{read_file, read_file_stream}` — `pub(crate)` entry points behind the two read trait
  methods; `prepare_read_stream`, `load_for_read`, `read_bytes_locked` are the private steps.
- `write::write_file` — `pub(crate)` entry point behind the write trait method;
  `validate_write`, `write_locked`, `version_conflict`, `apply_permissions` are private.
- `transfer::{upload_file, upload_file_stream, download_file, download_file_stream}` —
  `pub(crate)` entry points behind the four transfer trait methods; `prepare_upload`,
  `prepare_download`, `transfer`, `transfer_stream`, `copy_file`, `copy_chunks` are private.
- `local::{read_guard, write_guard, acquire_ordered}` are `pub(crate)` so the sibling modules
  can take locks; `local::{LockMode, PathGuard, len_u64}` are the shared lock/byte helpers.
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
- `local.rs` is the facade and the only place the lock map lives: the trait impls there are
  one-line delegations (`read::read_file(self, params).await`, `transfer::upload_file_stream(self,
  params)`, …), so behavior changes belong in the sibling module, never in the delegation.
- The sibling modules never touch the lock map directly: they call
  `LocalFsOperation::{read_guard, write_guard, acquire_ordered}`, which keep the ordering,
  poisoning-recovery, and mode-merge rules in one place.
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

- `local.rs` holds the facade, the lock map, and the shared fixtures only (~420 lines); read,
  write, and transfer logic each live in their own module next to their tests.
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

- 45 module-level unit tests live beside the code they exercise, all offline and deterministic
  (`tempfile` workspaces, no network, no host locale):
  `slice.rs` (validation order, line splitting, selection, chunk math, write-content composition),
  `read.rs` (read text/bytes/slicing/max-size/binary policy/traversal denial/symlink escape, text
  and byte stream shapes, the empty-file and tail terminal-chunk cases), `write.rs` (newline
  defaults, append semantics, flag-combination rejection, stale-digest and `create_if_absent`
  conflicts with a byte-identical file), `transfer.rs` (upload/download paths, parent creation,
  overwrite refusal, permissions, empty-source and multi-chunk stream shapes), `list.rs`,
  `search.rs` and `ext.rs` cover their own modules.
- `local.rs` keeps the delegation tests (`list_files`/`search_files`/ext ops) and the constructor
  tests, plus `#[cfg(test)] pub(crate) mod test_util` with the fixtures the sibling test modules
  import (`backend`, `restricted`, `canonical`, `read_back`); it holds no behavior tests of its own.
- Gates: `cargo fmt -p openjiuwen-sysop`, `cargo test -p openjiuwen-sysop --offline`,
  `cargo clippy -p openjiuwen-sysop --all-targets --offline -- -D warnings`,
  `cargo test -p openjiuwen-sysop --features sandbox --offline`.

## Routing Table

- `./mod.rs` → `local_fs` factory, module declarations (public types/list/search/ext/slice, crate-private read/write/transfer)
- `./types.rs` → frozen parameter/data types (`ReadFileParams`, `WriteFileParams`, `FileSystemItem`, ext types)
- `./local.rs` → `LocalFsOperation` facade, `new`/`policy`/`config`, per-path lock map (`read_guard`, `write_guard`, `acquire_ordered`, `LockMode`, `PathGuard`), `len_u64`, delegating trait impls, shared `test_util` fixtures
- `./read.rs` → `read_file`, `read_file_stream`, `prepare_read_stream`, `load_for_read`, `read_bytes_locked`, read stream plan types
- `./write.rs` → `write_file`, `validate_write`, `write_locked`, conditional compare-and-swap, permission application
- `./transfer.rs` → `upload_file`/`download_file` (+ streams), `prepare_upload`/`prepare_download`, ordered-lock copy engine, chunk streaming, host-path expansion
- `./slice.rs` → pure read validation/slicing, chunk sizes, write-content composition
- `./list.rs` → `list_files`, `list_directories`, directory walker, `file_system_item`, sorting
- `./search.rs` → `search_files` (glob + regex), exclude patterns
- `./ext.rs` → `stat`, `exists`, `mkdir`, `delete`, `copy`, `move_path`
