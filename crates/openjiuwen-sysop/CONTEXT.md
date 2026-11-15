# openjiuwen-sysop

## Intent

Filesystem, shell, and code operations behind one card, mirroring the Python package
`openjiuwen.core.sys_operation` (~45 files, ~8.2k lines) from openJiuwen v0.1.20.
A [`SysOperationCard`] selects a mode; the [`SysOperation`] facade hands out the matching
filesystem, shell, and code backends, so agent code is agnostic of where an operation runs.
Local mode executes in-process on tokio with cross-process file locks, process-group
lifecycle, optional PTY/`stdbuf` line buffering, and legacy CJK decoding and writing.
Sandbox mode forwards the same calls to a provider registered through
[`sandbox::provider`], resolving the target container through a launcher lifecycle.

## API Surface

### Facade — `sys_operation::{SysOperation, try_local}`

```rust
SysOperation::new(card: SysOperationCard) -> Result<Self>          // validates, builds backends
SysOperation::local(config: LocalWorkConfig) -> Result<Self>
SysOperation::sandbox(sandbox_type, gateway: SandboxGatewayConfig) -> Result<Self>
async SysOperation::sandbox_with_lifecycle(sandbox_type, gateway, session_id: Option<&str>,
                                           lifecycle: &SandboxLifecycle, now_epoch_secs: u64) -> Result<Self>
fn card(&self) -> &SysOperationCard
fn mode(&self) -> OperationMode
fn fs(&self) -> Option<Arc<dyn FsOperation>>       // Python `fs()` returns None for an unregistered kind
fn shell(&self) -> Option<Arc<dyn ShellOperation>>
fn code(&self) -> Option<Arc<dyn CodeOperation>>
fn isolation_key_template(&self) -> Option<&str>   // None in local mode
fn sandbox_endpoint(&self) -> Option<&SandboxEndpoint>          // Some only for a lifecycle-bound service
async fn release_sandbox(&self) -> Result<()>                   // applies launcher_config.on_stop; a launcher that cannot delete/pause refuses with `Unsupported`
async fn evict_idle_sandboxes(&self) -> Vec<String>              // deletes containers past idle_ttl_seconds
async fn evict_idle_sandboxes_at(&self, now_epoch_secs: u64) -> Vec<String>
fn tool_specs(&self) -> Vec<OperationToolSpec>     // 13 specs: 10 fs, 2 shell, 1 code
fn tool_specs_for(&self, kind: OperationKind) -> Vec<OperationToolSpec>
async fn read_file(&self, ReadFileParams) -> OpResult<ReadFileData>    // convenience
async fn execute_code(&self, CodeExecParams) -> OpResult<CodeOutput>   // convenience
```

`sandbox_with_lifecycle` resolves the endpoint through `SandboxLifecycle::ensure_endpoint`
before the provider backends are built; `sandbox` keeps the static endpoint (`base_url` from
the launcher config) and records no lifecycle binding. A lifecycle endpoint that carries no
isolation key is refused with 199010.

### Card and configuration — `card`, `config`, `mode`, `sandbox::config`

- `SysOperationCard { id, name, description, mode, work_config, gateway_config }`, Python JSON
  names, `mode` accepted case-insensitively, `id` defaults to `uuid_hex()`.
  `validate()` rejects an empty id and a sandbox card without launcher config (code 199002).
  `tool_id(kind, method)` = `"<card>.<kind>.<method>"`.
- `LocalWorkConfig`: Python fields `shell_allowlist`, `sandbox_root`, `restrict_to_sandbox`,
  `dangerous_patterns`; additions `workspace_root`, `max_file_size_bytes`, `max_output_bytes`,
  `default_timeout_secs`, `env_policy`, `cross_process_locks`, `lock_timeout_secs`,
  `line_buffered`, `pty_streaming`. Default allowlist = `constants::DEFAULT_SHELL_ALLOWLIST`
  (30 first tokens); `null`/`[]` means allow every command.
- `EnvPolicy { mode: All|None|Allowlist, allow, scrub }`, `BinaryPolicy::{Lossy, Reject}`.
- `SandboxGatewayConfig { isolation, launcher_config, timeout_seconds, auth_headers,
  auth_query_params, supports_conditional_write }`, `SandboxIsolationConfig`,
  `SandboxLauncherConfig` (single tolerant struct tagged by `launcher_type`, carrying
  `on_stop`, `idle_ttl_seconds`, `extra_params`, `base_url`), `ContainerScope`, `OnStop`,
  `SandboxRunConfig`, `GatewayInvokeRequest`, `GatewayResponse`, `SandboxEndpoint`.
- `generate_isolation_key_template(prefix, scope, custom_id, launcher_type, sandbox_type)` →
  `"{scope}_{launcher}_{sandbox}_{prefix}{identity}"`; `resolve_isolation_key(template, session)`
  replaces every `{session_id}` (default `default_session`).

### Operation traits — `operation`

`FsOperation` (10 Python methods + `supports_conditional_write`), `FsExtOperation` (Rust
additions: `stat`, `exists`, `mkdir`, `delete`, `copy`, `move_path`, `search_files_with`),
`ShellOperation` (`execute_cmd`, `execute_cmd_stream`, `execute_cmd_background`,
`execute_argv`), `CodeOperation` (`execute_code`, `execute_code_stream`).
Streaming methods take `self: Arc<Self>` and return `OpStream<T>` = `BoxStream<'static, OpResult<T>>`.
`OperationSet` bundles the three backends.

### Results — `result`, `status`

`OpResult<T> { code: i32, message: String, data: Option<T> }` is Python's `BaseResult[T]`
(`code == 0` on success; `data` omitted on failure). `OpResult::into_result()` converts to an
idiomatic `Result`; `into_optional_result()` keeps the payload optional.
`SysOpStatus` carries the Python codes: 0 success, -1 error, 199001 manager, 199002 card
param, 199003 fs, 199004 shell, 199005 code, 199006 registry, 199007 sandbox gateway,
199008 launcher, 199009 provider, 199010 isolation key, 199011 file version conflict, plus the
Rust-only 199012 (`SysOpStatus::Unsupported`: "the sandbox backend does not implement this
capability"; `OpResult::into_result`/`into_optional_result` turn it back into
`ErrorKind::Unsupported`).
Message builders (`fs_execution_message`, `shell_execution_message`, `code_execution_message`,
`registry_message`, `sandbox_gateway_message`, `sandbox_launcher_message`,
`sandbox_provider_message`, `isolation_key_message`, …) reproduce the Python templates
character for character.

### Types

`fs::types` — `ReadFileParams/Data`, `ReadChunkData`, `WriteFileParams/Data`,
`UploadFileParams/Data`, `DownloadFileParams/Data`, `TransferChunkData`, `ListFilesParams`,
`ListDirsParams`, `ListFilesData`, `FileSystemItem` (Python `type` field), `SearchFilesParams/Data`,
`SearchPatternMode`, `FileStat`, `MkdirParams/Data`, `DeleteParams/Data`, `CopyParams/Data`,
`MoveParams/Data`, `ReadMode`, `FileContent`, `LineRange`, `SortBy`.
`WriteFileParams.encoding` (`with_encoding`) selects the codec text content is written with;
an unknown encoding or an unrepresentable character is a 199003 failure.
`shell::types` — `ShellType`, `StreamKind`, `ShellExecParams`, `ShellArgvParams`, `ShellOutput`,
`ShellChunk`, `ShellBackgroundOutput`.
`code::types` — `Language`, `CodeExecParams`, `CodeOutput`, `CodeChunk`.
All parameter structs are `serde`-deserializable with Python defaults and derive `JsonSchema`,
so [`tool`] builds tool schemas and tool-call argument parsing from the same definition.

### Support modules

`workspace::PathPolicy` (resolution + traversal protection), `env`
(`ExecutionEnvironment`, `resolve_environment`), `util` (timestamps including pre-epoch ones,
octal permissions, SHA-256, UTF-8-safe truncation), `tool` (`OperationToolSpec` generation),
`constants`.
`encoding` — `TextEncoding` (utf-8, utf-16le/be, latin-1, gb18030, big5, shift-jis), every one
of them both decoded and written, `detect_text_encoding` (BOM → UTF-8 → UTF-16 heuristic →
strict GB18030/Big5 with Python's quality scoring → Latin-1), `decode_text`, `encode_text`,
`is_binary`, `bom_encoding`, `strip_bom`, `detected_encoding_name`; `encoding::cjk` adds
`CjkEncoding`, `decode`, `decode_strict`, and the streaming `CjkDecoder` over generated tables
in `encoding::tables`, and `encoding::encode` holds the Latin-1 encoder plus the lazy reverse
maps (`OnceLock<Vec<(char, Code)>>` + binary search) that make the CJK codecs writable.
`fs::slice::try_encoded_len` is the fallible byte-length probe behind the streamed text read;
`encoded_len` remains a bounded estimate.
`fs::lock` — `FileLockManager`, `FileLockGuard`, `LockMode`, `LockError`, `lock_path_for`,
`default_lock_root` (cross-process advisory `flock`).
`shell::pty` (crate-internal, Unix) — `openpty`-backed streaming spawn with a documented pipe fallback.
`sandbox::launcher` — `SandboxLauncher` trait, `LaunchedSandbox`, `SandboxStatus`
(`Running`/`Paused`/`Stopped`/`Unknown`), `LauncherRegistry`, `PreDeploymentLauncher`,
`global()`, error builders; only `launch` is required — the defaulted `launcher_type()` reports
`"custom"` (`PreDeploymentLauncher` reports `"pre_deploy"`), the inherited `pause`/`resume`/
`delete` refuse with `ErrorKind::Unsupported` naming the launcher, the operation and the
untouched container, and the inherited `check_status` reports `SandboxStatus::Unknown` instead
of claiming the container is running.
`sandbox::lifecycle` — `SandboxRecord` (carrying `container_config_hash` and `metadata`),
`Clock`/`SystemClock`/`FixedClock`, `container_config_hash`, `container_metadata`,
`refresh_container_config`, `SandboxLifecycle` (`ensure_endpoint`, `release`, `evict_idle`,
`evict_idle_now`, `evict_idle_for`, `record(s)`), `global()`.

## Constraints

- Errors: every backend call returns `OpResult<T>`; constructors return
  `openjiuwen_common::error::Result`. Failures never panic and never raise.
- Async-first: tokio I/O only, no blocking calls on an async task (heavy CPU work and the
  `flock` syscall use `spawn_blocking`); streaming is `BoxStream<'static, OpResult<T>>`.
- Public types are `serde`-serializable, tolerate unknown fields, and default missing
  collections; cards round-trip through JSON.
- `tracing` logging only, no `println!`; no `unsafe` (so no `pre_exec`, `setsid`, or
  `from_raw_fd` — see Notes for Agents); files ≤1000 lines, generated encoding tables excepted.
- Sandbox configuration types always compile; the HTTP client is behind the `sandbox` feature.
- Tests are offline and deterministic; they must not require `python3`/`node` to exist
  (skip with a message when the runtime is missing), must not touch the network, and must not
  depend on wall-clock ordering (lifecycle tests inject a clock).

## Python parity map

| Python (`core/sys_operation/…`) | Rust |
|---|---|
| `sys_operation.py` (`SysOperation`, `SysOperationCard`, `generate_isolation_key_template`) | `sys_operation.rs`, `card.rs`, `sandbox/config.rs` |
| `base.py` (`OperationMode`, `BaseOperation`) | `mode.rs`, `operation.rs` |
| `config.py` (`LocalWorkConfig`, sandbox models) | `config.rs`, `sandbox/config.rs` |
| `fs.py`, `shell.py`, `code.py` (ABCs) | `operation.rs` traits |
| `result/{fs,shell,code}_operation_result.py`, `result/base_result.py` | `fs::types`, `shell::types`, `code::types`, `result.rs` |
| `local/fs_operation.py` (1755 lines) | `fs/local.rs`, `fs/read.rs`, `fs/write.rs`, `fs/transfer.rs`, `fs/slice.rs`, `fs/list.rs`, `fs/search.rs`, `fs/ext.rs` |
| `local/shell_operation.py` (1049 lines) | `shell/local.rs`, `shell/allowlist.rs`, `shell/plan.rs`, `shell/process.rs`, `shell/pty.rs` |
| `local/code_operation.py` | `code/local.rs` |
| `local/utils.py` (`OperationUtils`, streaming decoding) | `env.rs`, `shell/process.rs`, `code/local.rs` |
| `local/_text_encoding.py`, `_rw_lock_manager.py`, `_async_read_write_lock.py` | `encoding/` (detection, `encode.rs` encoders over the generated CJK tables), `fs/lock.rs` (cross-process `flock`), `fs/local.rs` (in-process lock map) |
| `cwd.py` (`get_cwd`/`get_workspace`) | `workspace::PathPolicy` |
| `execution_environment.py` | `env.rs` |
| `shell_process_registry.py` | `shell/registry.rs` (process-group aware) |
| `tool_adapter.py` | `tool.rs` |
| `protocal/*.py`, `sandbox/sandbox_registry.py` | `sandbox::provider` (providers), `sandbox::launcher` (launchers) |
| `sandbox/{fs,shell,code}_operation.py`, `sandbox/gateway/*` | `sandbox::operation`, `sandbox::lifecycle`, `sandbox::http` |
| `sandbox/launchers/{base,pre_deployment_launcher}.py` | `sandbox::launcher` |
| `sandbox/providers/*`, `extensions/sys_operation` | external (`openjiuwen-extensions`); deferred transports |

## Behavior contract (ported from Python, verified against `tests/unit_tests/core/sys_operation`)

Read (`local/test_fs_operation.py`):
- `mode="bytes"` with any of `head`/`tail`/`line_range` → 199003
  `"only supported in text mode"`; only **non-zero** values participate.
- At most one of `head`/`tail`/`line_range` may be set → otherwise 199003
  `"cannot be specified simultaneously"`. Precedence when several are set: `tail` > `head` >
  `line_range`.
- `line_range` is 1-based and inclusive; `start<=0`, `end<=0`, `start>end`, or a range past
  EOF produce an **empty success** (Python returns `""`, not an error); `end` clamps to EOF.
- Negative `head`/`tail` → empty success. `head=0`/`tail=0` → full read.
- `chunk_size` applies to bytes mode only (`0` = whole file); our `ReadFileData.truncated`
  reports the silent truncation Python performs. Text mode ignores `chunk_size`.
- Text streaming emits one chunk per line; an empty file emits **zero** chunks; an invalid
  head/tail emits exactly one empty terminal chunk.
- `encoding="auto"` detects the encoding; BOM is stripped. Legacy CJK (`gbk`/`gb18030`,
  `big5`, `shift_jis` and their aliases) decodes with `U+FFFD` replacement, and every encoding
  this path decodes is also writable (see Write).

Write:
- `prepend_newline` defaults to **true** (a plain write gets a leading `\n`); `append_newline`
  defaults to false; `append` does not insert a newline and creates a missing file.
- `create_if_not_exist = false` on a missing file → 199003. `append` + missing file creates it.
- `expected_content_sha256` mismatch → `199011` with message `"file version conflict"` and the
  file left byte-identical; `create_if_absent` distinguishes an empty file from a missing one
  and rejects both when the file exists.
- `permissions` is applied best-effort on Unix as an octal string; not on Windows.
- `WriteFileParams.encoding` selects the codec of text content and every encoding
  `TextEncoding` parses is writable, so `decode_text(encode_text(s, encoding), encoding)`
  returns `s`: utf-8, utf-16le/be, latin-1 (true single-byte, `é` → `0xE9`), gb18030 (also
  `gbk`/`gb2312`/`cp936`, four-byte form and supplementary plane included), big5 (`cp950`,
  `big5-hkscs`) and shift-jis (`sjis`, `cp932`, two-byte area plus halfwidth katakana).
- Encoding failures surface as 199003: an unknown name and `auto` are `Error::config` (the
  message lists the writable encodings), an unrepresentable character is `Error::validation`
  (the message names the code point and the encoding).
- Text content is encoded before the target is opened, so a rejected encoding fails the write
  and leaves the target untouched, creating nothing.

Upload/download: `local_path` is **not** sandbox-checked (host side), `target_path`/
`source_path` are; existing destinations are rejected unless `overwrite`; parents are created
when `create_parent_dirs`.

List/search:
- `list_files`/`list_directories`: non-recursive = direct children; recursive without
  `max_depth` = full depth; `max_depth=0` = direct children only. `file_types` compares the
  suffix exactly (`".txt"`, case-sensitive). `sort_by` ∈ name/modified_time/size, optionally
  descending; a stat failure drops the entry instead of failing the call. Recursion does not
  follow directory symlinks.
- `search_files` is **glob** matching (Python `pathlib.rglob`: `*`, `?`, `[]`, `**`), returns
  files only, applies `exclude_patterns` as globs, and treats no match as success with
  `total_matches == 0`. `search_files_with(..., SearchPatternMode::Regex)` is the additive
  regex variant.
- A walk that fails at its root is a 199003 failure, never an empty success; an unreadable
  subdirectory or an entry whose `stat` fails is skipped and logged at `warn!` (Python
  `os.walk` parity).

Shell (`local/test_shell_operation.py`):
- Empty/blank command → 199004 `"command can not be empty"`, `exit_code = -1`.
- Timeout: `timeout_secs` `None`/`0` → configured default (300). Expiry kills the **process
  group**, keeps partial output, sets `timed_out`, `exit_code = -1`, and returns 199004
  containing `"timeout"`.
- Allowlist: `None` or `[]` allows every command; otherwise a command is allowed when its
  first token's normalized leaf name equals an entry or ends with `"/<entry>"` (Windows also
  drops a leading `&`, normalizes `\`→`/`, casefolds, and strips one trailing `.exe`).
  Rejection → 199004 `"not allowed"`.
- Deny list: `dangerous_patterns` regexes (case-insensitive) replace the built-in list, which
  is **empty** in Python; `config::DEFAULT_DANGEROUS_PATTERNS` is an opt-in table.
- `cwd` resolves against the workspace root; with `restrict_to_sandbox` both the cwd and
  absolute path literals in the command must stay inside the roots.
- Output capture is bounded by `max_output_bytes` per stream with
  `constants::OUTPUT_TRUNCATION_MARKER`; `ShellOutput.truncated` reports it. Python captures
  without a limit (documented deviation).
- Environment: policy-scoped inheritance (`EnvPolicy`), explicit overlay wins, `None`
  removes. Python inherits everything and never scrubs.
- Streaming: chunks carry `type` (stdout/stderr) and `chunk_index`; the terminal chunk carries
  `exit_code`; a failure yields one error chunk.
- Background: returns `pid`; an early non-zero exit yields
  `"process exited early with code <n>: <stderr>"` (stderr retained up to 8192 bytes), and the
  tracked process group can be stopped through `shell::registry`.

Code (`local/test_code_operation.py`):
- Empty code → 199005 `"code can not be empty"`; unsupported language → 199005
  `"<lang> is not supported"`; missing runtime → 199005
  `"<lang> file not found error, please install and add it to your system environment variable PATH."`.
- A **non-zero exit is not an error**: the result is code 0 with
  `message = "Code executed successfully"` and `exit_code != 0`.
- Snippets longer than `UNIX_COMMAND_LINE_LIMIT` (100k) / `WINDOWS_COMMAND_LINE_LIMIT` (8k), or
  `force_file`, run from a temporary file that is deleted afterwards (also on timeout).
- Forced environment after user overrides: Python `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`;
  JavaScript `NODE_DISABLE_COLORS=1`.
- Timeout → 199005 with partial output and `timed_out = true`.

Cross-process locks (`_rw_lock_manager.py`):
- Two layers: the in-process `tokio::sync::RwLock` map (fast path, shared for reads,
  exclusive for writes) then an advisory `flock` on a sidecar file, acquired in one canonical
  order (resolved path bytes, duplicates merged with `Write` winning) and released in reverse.
- Sidecar layout: `<default_lock_root>/<sha256-hex-of-the-absolute-resolved-target>.lock`,
  where `default_lock_root()` is `$TMPDIR/openjiuwen-sysop-locks`; a host may point the
  manager elsewhere with `FileLockManager::with_root`.
- `lock_timeout_secs` (default 30, `0` = wait forever) bounds the wait; expiry is `199003`
  with the method name, the path, and a timeout reason, and **never** executes the operation
  unlocked. A lock file that cannot be created also refuses the operation.
- `cross_process_locks = false` restores single-layer in-process locking; reads with
  `only_read = true` take no lock at all (Python `skip_lock`).
- The advisory lock is per open file description, so it serialises threads of one process
  (separate descriptors) and distinct processes alike; it is not reentrant — a second
  acquisition of a path already held by the same operation is merged, never nested.

Process groups and buffering (`shell_operation.py`, `shell_process_registry.py`):
- Every child starts in its own process group (`Command::process_group(0)`); timeout,
  cancellation-token fire, and background stop terminate the **group** with `SIGKILL`
  (`killpg`), then reap the direct child exactly once. A group is never signalled after its
  child was reaped (pid reuse), and never with a non-positive pid.
- `line_buffered` (default true) prefixes `execute_cmd`/`execute_cmd_stream`/
  `execute_cmd_background` with `stdbuf -oL -eL` when `stdbuf` resolves; `execute_argv` is
  never wrapped. A missing `stdbuf`, or a non-Unix host, runs the command unwrapped rather
  than failing it.
- `pty_streaming` (default false) attaches *streaming* commands to a pseudo-terminal so the
  child line-buffers: `openpty`, `ECHO`/`OPOST`/`ONLCR` cleared, slave on stdin/stdout/stderr,
  master read asynchronously, `\r` dropped. All PTY chunks are `StreamKind::Stdout` (a
  terminal merges stderr), and a PTY that cannot be allocated falls back to pipes silently.
  Captured `execute_cmd` always uses pipes, keeping stdout and stderr separate.
- Teardown stays observable: a failed group kill, a reaping failure, the PTY→pipe fallback, a
  reader error and an unusable exit status are logged instead of being dropped.

Sandbox lifecycle (`sandbox/gateway/gateway.py`, `sandbox/launchers/`):
- `SandboxLifecycle::ensure_endpoint` reuses a running record (refreshing `last_used` and its
  configuration bookkeeping), resumes a paused one, relaunches a stopped one, evicts idle
  records when `idle_ttl_seconds` is set, and stores the new record. The map mutex is never
  held across a launcher await; a concurrent launch keeps the first record and deletes the
  duplicate.
- `release(isolation_key, on_stop)`: `Delete` destroys, `Pause` suspends and keeps the record
  (so the next lookup resumes that container), `Keep` only refreshes `last_used`; unknown keys
  are a no-op.
- A launcher that cannot perform the call reports `ErrorKind::Unsupported` instead of a
  success it did not achieve; the record is restored on every failure, so a refused teardown
  is retryable and keeps its `container_config_hash`/`metadata`. Only a real `SysOp` launcher
  failure becomes 199008.
- `evict_idle(ttl, now)` deletes every record idle for at least `ttl` regardless of `on_stop`,
  drops the record and returns its key even when the launcher cannot delete the container: the
  container is logged as left behind at `warn!` with `launcher_type`/`sandbox_id`, and the
  sweep continues to the remaining records. All time is injected (`Clock`, `FixedClock` in
  tests), and sweeps are caller-driven — `ensure_endpoint` when `idle_ttl_seconds` is set, or
  an explicit `evict_idle`/`evict_idle_now`/`evict_idle_for` call; there is no background task.
- `SandboxRecord.container_config_hash` is the SHA-256 of the launcher config's canonical JSON
  (`container_config_hash`) and `metadata` carries `launcher_type`, `sandbox_type` and the
  string-valued `extra_params` (`container_metadata`); `refresh_container_config` merges both
  after a successful launcher call on reuse and resume, keeps unrelated labels and logs a
  changed hash at `debug` because a provider cache keyed by it would be invalidated.
- Error mapping: launcher failures 199008, provider failures 199009, isolation-key problems
  199010, gateway failures 199007 — all `ErrorKind::SysOp`. `map_launcher_failure` passes
  `ErrorKind::Unsupported`/`Validation`/`Config` through unchanged, because they say what the
  launcher is, not that a call failed. `ErrorKind::Unsupported` is also the answer for a
  genuinely absent transport: an unregistered `launcher_type`, an unregistered `sandbox_type`,
  or the `sandbox` feature being off.

## Security Model

- **Path containment**: every path goes through `workspace::PathPolicy`. `..` is removed
  lexically before any I/O, symbolic links are resolved for the longest existing prefix, and
  when `restrict_to_sandbox` is set the resolved path must be equal to or below one of the
  roots (`is_within` is component-wise, so `/root-other` is not inside `/root`). Violations
  return 199003 with `"Access denied: Path ... outside sandbox [...]"`.
- **Writes** always create parent directories inside the sandbox only; the leaf is opened
  after resolution, so a symlinked leaf cannot be redirected outside.
- **Command execution**: `execute_cmd` interprets shell syntax and is therefore gated by the
  first-token allowlist plus the optional regex deny list; `execute_argv` never uses a shell,
  which is the supported way to run a command whose arguments come from untrusted input.
  Independent of the allowlist, no code path interpolates user input into a shell string.
- **Resource limits**: `max_output_bytes` bounds capture; `max_file_size_bytes` bounds reads;
  `default_timeout_secs` bounds runtime; `max_depth` bounds traversal. Processes are killed as
  a group on timeout and on cancellation through `shell::registry`.
- **Environment**: `EnvPolicy` can switch from Python's inherit-everything default to a
  minimal allowlist and can scrub secrets; explicit values can never re-add a scrubbed name.
- **No `unsafe`**, no panics on malformed paths/commands, no `unwrap` outside tests.
- **Concurrency**: same-process consistency comes from the per-path `RwLock` map;
  cross-process consistency from the advisory `flock` sidecar (a cooperative lock: another
  program that ignores it can still write). Cross-process *reads* are shared, writes
  exclusive, and a transfer locks both endpoints in canonical order, so two processes cannot
  interleave a read with a write of the same path.

## Known Issues / Deferred

- Remote sandbox transports: `sandbox::launcher` (`SandboxLauncher`, `LauncherRegistry`,
  built-in `PreDeploymentLauncher`) and `sandbox::lifecycle` (`ensure_endpoint` reuse/resume/
  launch, `release` by `OnStop`, `evict_idle` via an injected clock) manage containers per
  isolation key, exposed through `SysOperation::sandbox_with_lifecycle`,
  `release_sandbox`, `evict_idle_sandboxes*`. The `aio`/`jiuwenbox`/`yuanrong` transports and
  their launchers live in `openjiuwen-extensions` and are not ported here: no
  provider/launcher registered for a type, or the `sandbox` feature off, yields
  `Error::Unsupported`; a registered launcher that fails yields 199008. Still absent: the
  provider cache Python keys by `container_config_hash` (this rewrite rebuilds providers from
  the resolved endpoint), endpoint discovery beyond `LaunchedSandbox`, auth-header/auth-query
  injection, the streaming wire format (`invoke` stays `Unsupported`, surfaced as code 199012),
  and any background idle sweeper (idle-stop runs on `ensure_endpoint` when
  `idle_ttl_seconds` is set, or on an explicit `evict_idle_sandboxes*` call). Deviations:
  `release(Pause)` keeps the record `paused` so the next lookup resumes the same container
  (Python forgets it), and `evict_idle` compares `>=` (Python `>`).
- Legacy CJK is table driven in both directions — `encoding::cjk` over the generated
  `encoding::tables` decodes, `encoding::encode` builds the reverse maps — and covers
  GB18030/GBK, Big5 and Shift-JIS with `U+FFFD` replacement on decode. Gaps by design: no
  Big5-HKSCS or CP932 vendor extensions (their alias names accept the standard code space); no
  write aliases beyond the mapped code space; Big5's duplicate mappings pick the smallest code
  (`十` → `0xA2CC`, where Python picks `0xA451`); Shift-JIS refuses the vendor single-byte `¥`,
  whose byte `0x5C` decodes as `\`; Shift-JIS is never auto-detected (Python does not attempt
  it either); and `auto` detection strict-decodes the whole in-memory buffer where Python
  samples files above 8 MiB.
- Writes cover the seven supported encodings (utf-8, utf-16le/be, latin-1, gb18030, big5,
  shift-jis); Python can write arbitrary codecs.
- `PreDeploymentLauncher` owns no handle on a shared-scope container, so it genuinely cannot
  manage one: `pause`/`resume`/`delete` return `ErrorKind::Unsupported` naming the launcher,
  the operation and the untouched container, `check_status` reports `Unknown`, and
  `SandboxLifecycle::release(Delete)` therefore fails and restores the record instead of
  reporting a teardown. Python's `pre_deployment_launcher.delete` deletes
  `jiuwenbox`/`yuanrong` containers through the extension providers.
- `SandboxLifecycle::evict_idle` drops the record and reports the key as evicted even when the
  launcher cannot delete the container or its type is no longer registered, so that container
  is leaked (logged at `warn` with `launcher_type`/`sandbox_id`); a duplicate launch whose
  cleanup delete fails is leaked the same way.
- The HTTP gateway reports a capability gap as code `199012` (`SysOpStatus::Unsupported`)
  rather than 199007: `gateway_failure` maps `ErrorKind::Unsupported` through
  `result::unsupported_failure`/`status::unsupported_message` in the same `sandbox gateway
  error, operation: <method>, reason: <reason>` envelope it uses for every other kind, a
  streaming call yields exactly one error chunk with that code, and `OpResult::into_result`/
  `into_optional_result` turn 199012 back into `ErrorKind::Unsupported`. Serialization
  failures, a non-zero `response.code` and a missing or undecodable payload stay 199007.
- Hard links to one inode map to different sidecar lock files (Python's SQLite lock manager
  shares that limitation), and stale lock files are never swept (they are empty and reused).
- POSIX streaming does not reproduce the macOS `script -q /dev/null` wrapper; macOS runs
  unwrapped, and the PTY path is the supported way to get terminal semantics there.
- Background early-exit detection uses a 100 ms grace window (Python defaults to 3 s and
  exposes `grace=`), so a command failing later than that is reported as started, with a pid.
- `execute_argv` applies the allowlist to `argv[0]` but not `dangerous_patterns`: arguments are
  data, never shell text.
- Windows `Auto` uses `cmd /C`; Python's POSIX/PowerShell command classification heuristics and
  `-lc` path normalization are not ported.
- Shell and code output capture is bounded by `max_output_bytes` (Python captures unbounded);
  timeout and cancellation failures keep partial output in `data`.

## Test Strategy

- `#[cfg(test)]` modules next to the code cover unit behavior: parameter defaults, slicing,
  encoding detection and writing (round-trips, byte fixtures, unknown and unrepresentable
  names), allowlist parsing, path containment, truncation, result mapping, launcher refusals,
  record `container_config_hash`/`metadata` bookkeeping, clock-driven idle sweeps, the lifecycle
  `Debug` locked marker, unreadable walk roots, and pre-epoch timestamps.
- Shell and code behavior ports live in `#[cfg(test)]` modules beside the backends:
  allowlist acceptance/rejection, deny patterns, timeout with partial output, bounded capture,
  env overlay/scrubbing, `execute_argv` without a shell, background early exit, stream chunk
  shape, session/instance cancellation, the `stdbuf` wrapper decision, the process-group
  kill-once guard, PTY fallback and `\r` stripping, code language metadata and temp-file
  strategy.
- `tests/` holds cross-module behavior ports, each citing the Python test it ports:
  `card_json.rs` (14), `local_fs.rs` (18), `fs_edges.rs` (20), `traversal.rs` (12),
  `shell_local.rs` (17), `code_local.rs` (10), `tool_specs.rs` (9), `encoding_cjk.rs` (17
  offline GB18030/Big5/Shift-JIS fixtures), `encoding_write.rs` (9: round-trips for every
  encoding, known byte fixtures, unknown and unrepresentable failures), `file_locks.rs` (10:
  contention, timeout, and a genuine second-process proof by re-executing the test binary),
  `process_group.rs` (5: timeout/cancel/background-stop leave no grandchild),
  `pty_streaming.rs` (5: terminal detection, line-buffered chunks, exit codes),
  `sandbox_lifecycle.rs` (27: registry, reuse/resume/release/idle-stop, isolation keys, config
  hash/metadata, `aio`/`jiuwenbox`/`yuanrong` interop with local registries),
  `sandbox_stub.rs` (14, feature `sandbox`: unsupported remote calls, isolation keys, provider
  dispatch), and `sandbox_gateway_errors.rs` (5, feature `sandbox`: 199012 for a capability gap
  against 199007 for a real gateway failure, unary and streaming).
- `src/sandbox/http.rs` and `src/sandbox/operation.rs` keep their tests inline: fake-provider
  forwarding (exact `OpResult` and stream passthrough), dispatch + per-session isolation-key
  resolution, missing provider (`ErrorKind::Unsupported`) and missing gateway/launcher config
  (199002) paths, `install` idempotency, and one `#[ignore]`d live `probe` test.
- Every test is offline and deterministic: `tempfile` workspaces, no network, injected clocks
  for lifecycle tests, bounded waits, and runtime lookups (`python3`, `node`, `bash`, `stdbuf`,
  `flock(1)`) skip with an explanatory message when missing.
- Gates: `cargo fmt -p openjiuwen-sysop`, `cargo test -p openjiuwen-sysop`,
  `cargo test -p openjiuwen-sysop --features sandbox`,
  `cargo clippy -p openjiuwen-sysop --all-targets -- -D warnings` (both feature sets).
- Current counts: 343 unit tests (355 with `sandbox`, 1 `#[ignore]`d live probe), 173
  integration tests across 13 files (192 across 15 with `sandbox`), 35 doc tests.

## Routing Table

- `./src/sys_operation.rs` → `SysOperation` facade, mode dispatch, lifecycle binding, tool spec aggregation
- `./src/card.rs` → `SysOperationCard`, validation, tool-id generation
- `./src/config.rs` → `LocalWorkConfig`, `EnvPolicy`, `BinaryPolicy`, `DEFAULT_DANGEROUS_PATTERNS`
- `./src/mode.rs` → `OperationMode`, `OperationKind`
- `./src/status.rs` → `SysOpStatus`, Python message templates
- `./src/result.rs` → `OpResult`, failure builders, `Result` conversion
- `./src/operation.rs` → `FsOperation`, `FsExtOperation`, `ShellOperation`, `CodeOperation`, `OpStream`, `OperationSet`
- `./src/workspace.rs` → `PathPolicy`, path cleaning, symlink resolution, containment checks
- `./src/encoding/` → `TextEncoding`, detection/BOM/binary sniff/`decode_text`/`encode_text`, plus the `cjk` decoders, the `encode` reverse maps and generated `tables`; module detail in `src/encoding/CONTEXT.md`
- `./src/env.rs` → `ExecutionEnvironment`, `resolve_environment`, env-name validation
- `./src/util.rs` → timestamp formatting, octal permissions, SHA-256, UTF-8-safe truncation
- `./src/tool.rs` → `OperationToolSpec` generation from parameter schemas
- `./src/constants.rs` → defaults shared by the backends
- `./src/fs/` → filesystem types and the local backend (`local`, `read`, `write`, `transfer`, `slice`, `list`, `search`, `ext`) plus `lock` (cross-process `flock`); module layout in `src/fs/CONTEXT.md`
- `./src/shell/types.rs` → `ShellType`, `ShellExecParams`, `ShellArgvParams`, `ShellOutput`, `ShellChunk`, `ShellBackgroundOutput`
- `./src/shell/allowlist.rs` → `CommandFilter`: first-token allowlist (quote-aware splitter) and dangerous-pattern deny list
- `./src/shell/plan.rs` → `resolve_plan` per `ShellType` and OS, absolute-path literal extraction
- `./src/shell/registry.rs` → `ShellProcessRegistry`: per-session cancellation tokens, tracked process groups, RAII registration, `global()`
- `./src/shell/process.rs` → shared child-process engine: hardened spawn, `stdbuf` wrapper, bounded capture + truncation, timeout/cancellation, process-group kill, streaming, background stderr
- `./src/shell/pty.rs` → `openpty` streaming spawn (termios setup, master reader, pipe fallback)
- `./src/shell/local.rs` → `LocalShellOperation`: policy gates, spawn plan, trait impls, facade tests
- `./src/code/types.rs` → `Language`, `CodeExecParams`, `CodeOutput`, `CodeChunk`
- `./src/code/local.rs` → `LocalCodeOperation`: runtime lookup, inline/temp-file strategy, forced env
- `./src/sandbox/` → provider seam, launcher seam + lifecycle, gateway config, dispatch + forwarding backends, HTTP gateway client (feature `sandbox`); module detail in `src/sandbox/CONTEXT.md`
- `./scripts/gen_cjk_tables.py` → authoring-time, self-verifying CJK table generator (`--check` in review); never needed by a build or test
- `./tests/` → behavior ports (see Test Strategy)

## Notes for Agents

- Every public item needs a doc comment (`missing_docs` is warn and CI uses `-D warnings`);
  clippy runs with `all` + `pedantic`, so prefer `#[must_use]`, `# Errors` sections on
  fallible functions, and no `unwrap`/`expect` outside `#[cfg(test)]`.
- Failure messages must go through `crate::status` builders so codes and text stay
  Python-compatible; the `execution` slot is the method name (`"read_file"`, `"execute_cmd"`).
- Never bypass `PathPolicy` when touching the filesystem: a raw `std::fs` call on a
  caller-supplied path is a security bug.
- `FileContent` decides text vs bytes on write; there is no separate `mode` field, so a
  mismatch cannot occur.
- Take locks only through `LocalFsOperation::{read_guard, write_guard, acquire_ordered}`: they
  own the single ordering rule (sorted resolved paths, in-process guard then `flock`) and the
  reverse-release rule. Never take a bare `flock` in a sibling module.
- `nix::fcntl::flock` is deprecated in nix 0.30 in favour of `Flock`, whose `Drop` panics on
  unlock failure; the crate deliberately keeps the function behind a scoped
  `#[allow(deprecated)]` and owns the `std::fs::File` instead. `killpg` and `openpty` use the
  supported safe APIs, and every fd conversion goes through `Stdio::from(OwnedFd)` /
  `File::from(OwnedFd)` because `#![forbid(unsafe_code)]` rules out `pre_exec`/`setsid` and
  `from_raw_fd`.
- Process-group rule: kill the group only *before* reaping the direct child (an unreaped
  leader keeps the pgid addressable), never after, never with pid 0, and at most once.
- The `stdbuf` wrapper is applied at spawn time, never in `resolve_plan`, so allowlist and
  sandbox path checks always see the command the caller wrote, and the plan unit tests keep
  asserting the raw interpreter.
- A PTY merges stderr into stdout: never route captured runs or `execute_argv` through
  `shell::pty`, and keep `pty_streaming = false` the default.
- `sandbox::operation` resolves everything in a private core that takes a
  `&SandboxProviderRegistry`: tests exercise dispatch against a local registry and never
  register mocks in `sandbox::provider::global()` (only `SysOperation` uses the global one).
- Same rule for launchers: `LauncherRegistry::new()` in tests, `sandbox::launcher::global()`
  only in production paths; `register_builtin_launchers`/`register_builtin_providers` are
  idempotent so repeated `SysOperation::new` calls never fail.
- `SandboxLifecycle` never holds its map mutex across a launcher await, and all time is passed
  in (`now_epoch_secs` or a `Clock`) so idle-stop is deterministic in tests.
- Absence of state is never reported as empty state: a poisoned launcher/provider registry lock
  is logged and renders as `["<poisoned>"]`, and a lifecycle `Debug` whose record map is busy
  renders `("<locked>", Unknown)`; a launcher refusal is a typed error, never an `Ok(())`.
- Sandbox forwarding backends (`SandboxFsOperation`, `SandboxShellOperation`,
  `SandboxCodeOperation`) never serialize, inspect, or re-map provider results: the provider
  owns the wire format. `supports_conditional_write` returns the provider's answer; the card's
  claim is exposed separately as `configured_conditional_write`.
- Streams must be `'static`; use `self: Arc<Self>` receivers and `async_stream::try_stream!`,
  yielding a single error chunk on failure instead of returning `Err`.
- The local shell and code backends share `crate::shell::process` (`spawn_child`,
  `apply_environment`, `run_captured`, `stream_child`, `line_buffered_argv`, `ProcessGroup`);
  never re-implement bounded capture, timeout, cancellation, or group kill inside
  `code/local.rs`.
- fs `read_file`/`write_file`/upload/download live in `src/fs/{read,write,transfer}.rs`; keep the
  trait impls in `fs/local.rs` one-line delegations and share the single per-path lock map.
- Shell gate order is blank command → deny list → allowlist → cwd/sandbox → shell plan, so a
  rejected command never spawns a child; `CommandFilter::check` reports the deny list first.
- Encoding tables are generated: edit `scripts/gen_cjk_tables.py` and regenerate, never the
  `encoding/tables/*.rs` data. `TextEncoding::as_str` reports the covering canonical name, so
  `gbk` reports `gb18030`; `decode_text` accepts the alias but the reported name does not
  round-trip.
- `<lang> is not supported` is unrepresentable here (`Language` is a closed enum), and
  `CodeExecParams::encoding` is accepted but ignored: decoding is UTF-8 lossy.
