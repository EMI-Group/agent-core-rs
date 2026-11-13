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
  can take locks; each returns `Result<PathGuard, String>` (the `String` is the 199003 reason) and
  takes both lock layers. `local::{LockMode, PathGuard, len_u64}` are the shared lock/byte
  helpers; `LockMode` is re-exported from `lock.rs`, which owns the single definition.
- `lock::{FileLockManager, FileLockGuard, LockMode, LockError, lock_path_for,
  default_lock_root, LOCK_DIR_NAME, POLL_INTERVAL}` — the cross-process advisory `flock` layer
  (Unix; a no-op guard elsewhere). `FileLockManager` is cheaply cloneable and exposes
  `new`/`with_root`/`with_timeout`, `is_enabled`, `timeout`, `lock_path`, `try_acquire` and
  `acquire`.
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
- The sibling modules never touch the lock maps directly: they call
  `LocalFsOperation::{read_guard, write_guard, acquire_ordered}`, which keep the ordering,
  poisoning-recovery, mode-merge, and two-layer acquisition rules in one place.
- Failures are `OpResult` values (`fs_failure(execution, reason)`, code 199003; code 199011 with
  message `file version conflict` for a rejected conditional write). Nothing panics.
- Blocking work stays off the async task: `walkdir` walks run in `spawn_blocking`; file I/O is
  `tokio::fs`; the `flock` syscall runs in `spawn_blocking`; only cheap
  `set_permissions`/metadata calls on already-resolved paths are sync.
- Locking is two-layered *and ordered*. Layer 1 is a per-resolved-path `tokio::sync::RwLock` in a
  `std::sync::Mutex<HashMap<..>>` (the std guard is never held across an `await`, and a poisoned
  mutex is recovered, never unwrapped). Layer 2 is the advisory `flock` from `lock.rs`, taken
  after layer 1 for the same path. Readers take the shared mode (both layers skipped when
  `only_read`), writers the exclusive one, and transfers lock both paths sorted by path (`Write`
  wins) so transfers cannot deadlock in either layer. `cross_process_locks = false` leaves only
  layer 1, with no lock files created.
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
- Lock-map entries and lock files are never evicted; memory grows with the number of distinct
  resolved paths a process touches, and the `openjiuwen-sysop-locks` temp directory keeps one
  empty `.lock` file per path ever locked (Python's `ReadWriteLockManager` runs a cleanup task;
  this port leaves reclamation to the OS temp cleaner).
- Cross-process locks only defend against other processes that also use these helpers; a program
  writing the file directly does not take the `flock`.
- The `flock` sidecar is per resolved path, so two hard links to one inode lock different files
  (Python's SQLite lock manager has the same limitation).
- `ext::{stat, exists, mkdir, delete, copy, move_path}` never took a lock (they predate the lock
  map and have no `LocalFsOperation` handle), and they still do not; only `read_file`,
  `write_file`, and the four transfer methods lock.
- Non-Unix platforms (and `cross_process_locks = false`) yield an inactive guard immediately, so
  only the in-process layer applies.
- Directory copies skip symlinks found inside the tree (logged at `debug`) so a link cannot pull
  content from outside the sandbox.

## Test Strategy

- 53 module-level unit tests live beside the code they exercise, all offline and deterministic
  (`tempfile` workspaces, no network, no host locale):
  `slice.rs` (validation order, line splitting, selection, chunk math, write-content composition),
  `read.rs` (read text/bytes/slicing/max-size/binary policy/traversal denial/symlink escape, text
  and byte stream shapes, the empty-file and tail-terminal-chunk cases), `write.rs` (newline
  defaults, append semantics, flag-combination rejection, stale-digest and `create_if_absent`
  conflicts with a byte-identical file), `transfer.rs` (upload/download paths, parent creation,
  overwrite refusal, permissions, empty-source and multi-chunk stream shapes), `lock.rs`
  (digest naming, config -> enablement/timeout mapping, shared/exclusive compatibility through
  separate descriptors, timeout reason text, disabled manager creating nothing, guard `Send` +
  release on drop), `list.rs`, `search.rs` and `ext.rs` cover their own modules.
- `local.rs` keeps only the delegation tests (`list_files`/`search_files`/ext ops) and the
  constructor tests, plus `#[cfg(test)] pub(crate) mod test_util` with the fixtures the sibling test
  modules import (`backend`, `restricted`, `canonical`, `read_back`).
- `tests/file_locks.rs` proves the cross-process layer with 10 tests: same-process contention and
  `try_acquire`, genuine second-process contention (the test binary re-execs itself as a helper via
  `SYSOP_LOCK_HELPER` and exits 3 while the parent holds the lock, 0 after the drop), the optional
  util-linux `flock -n` probe (skipped when absent), the two 199003 timeout mappings
  (`read_file`/`write_file` with the `execution` slot), reversed upload/download deadlock freedom
  (plain and streaming, under a 10 s bound), and the `cross_process_locks` switch creating or not
  creating the sidecar.
- Gates: `cargo fmt -p openjiuwen-sysop`, `cargo test -p openjiuwen-sysop --offline`,
  `cargo clippy -p openjiuwen-sysop --all-targets --offline -- -D warnings`,
  `cargo test -p openjiuwen-sysop --features sandbox --offline`,
  `cargo clippy -p openjiuwen-sysop --all-targets --features sandbox --offline -- -D warnings`.
  Current counts: 245 unit tests without the feature / 253 with it (1 `#[ignore]`d live sandbox
  probe), 109 integration tests without the feature / 123 with it, 29 doc tests.

## Routing Table

- `./mod.rs` → `local_fs` factory, module declarations (public types/list/search/ext/slice/lock, crate-private read/write/transfer)
- `./types.rs` → frozen parameter/data types (`ReadFileParams`, `WriteFileParams`, `FileSystemItem`, ext types)
- `./local.rs` → `LocalFsOperation` facade, `new`/`policy`/`config`, two-layer lock helpers (`read_guard`, `write_guard`, `acquire_ordered`, `PathGuard`, re-exported `LockMode`), `len_u64`, delegating trait impls, shared `test_util` fixtures
- `./lock.rs` → cross-process advisory `flock`: `FileLockManager`, `FileLockGuard`, `LockMode`, `LockError`, `lock_path_for`, `default_lock_root`, polling and timeout, non-Unix no-op fallback
- `./read.rs` → `read_file`, `read_file_stream`, `prepare_read_stream`, `load_for_read`, `read_bytes_locked`, read stream plan types
- `./write.rs` → `write_file`, `validate_write`, `write_locked`, conditional compare-and-swap, permission application
- `./transfer.rs` → `upload_file`/`download_file` (+ streams), `prepare_upload`/`prepare_download`, ordered-lock copy engine, chunk streaming, host-path expansion
- `./slice.rs` → pure read validation/slicing, chunk sizes, write-content composition
- `./list.rs` → `list_files`, `list_directories`, directory walker, `file_system_item`, sorting
- `./search.rs` → `search_files` (glob + regex), exclude patterns
- `./ext.rs` → `stat`, `exists`, `mkdir`, `delete`, `copy`, `move_path`
