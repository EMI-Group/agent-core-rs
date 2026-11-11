# openjiuwen-sysop

## Intent

Filesystem, shell, and code operations behind one card, mirroring the Python package
`openjiuwen.core.sys_operation` (~45 files, ~8.2k lines) from openJiuwen v0.1.20.
A [`SysOperationCard`] selects a mode; the [`SysOperation`] facade hands out the matching
filesystem, shell, and code backends, so agent code is agnostic of where an operation runs.
Local mode executes in-process on tokio; sandbox mode forwards the same calls to a provider
registered through [`sandbox::provider`].

## API Surface

### Facade — `sys_operation::{SysOperation, try_local}`

```rust
SysOperation::new(card: SysOperationCard) -> Result<Self>          // validates, builds backends
SysOperation::local(config: LocalWorkConfig) -> Result<Self>
SysOperation::sandbox(sandbox_type, gateway: SandboxGatewayConfig) -> Result<Self>
fn card(&self) -> &SysOperationCard
fn mode(&self) -> OperationMode
fn fs(&self) -> Option<Arc<dyn FsOperation>>       // Python `fs()` returns None for an unregistered kind
fn shell(&self) -> Option<Arc<dyn ShellOperation>>
fn code(&self) -> Option<Arc<dyn CodeOperation>>
fn isolation_key_template(&self) -> Option<&str>   // None in local mode
fn tool_specs(&self) -> Vec<OperationToolSpec>     // 13 specs: 10 fs, 2 shell, 1 code
fn tool_specs_for(&self, kind: OperationKind) -> Vec<OperationToolSpec>
async fn read_file(&self, ReadFileParams) -> OpResult<ReadFileData>    // convenience
async fn execute_code(&self, CodeExecParams) -> OpResult<CodeOutput>   // convenience
```

### Card and configuration — `card`, `config`, `mode`, `sandbox::config`

- `SysOperationCard { id, name, description, mode, work_config, gateway_config }`, Python JSON
  names, `mode` accepted case-insensitively, `id` defaults to `uuid_hex()`.
  `validate()` rejects an empty id and a sandbox card without launcher config (code 199002).
  `tool_id(kind, method)` = `"<card>.<kind>.<method>"`.
- `LocalWorkConfig`: Python fields `shell_allowlist`, `sandbox_root`, `restrict_to_sandbox`,
  `dangerous_patterns`; additions `workspace_root`, `max_file_size_bytes`, `max_output_bytes`,
  `default_timeout_secs`, `env_policy`. Default allowlist = `constants::DEFAULT_SHELL_ALLOWLIST`
  (30 first tokens); `null`/`[]` means allow every command.
- `EnvPolicy { mode: All|None|Allowlist, allow, scrub }`, `BinaryPolicy::{Lossy, Reject}`.
- `SandboxGatewayConfig { isolation, launcher_config, timeout_seconds, auth_headers,
  auth_query_params, supports_conditional_write }`, `SandboxIsolationConfig`,
  `SandboxLauncherConfig` (single tolerant struct tagged by `launcher_type`),
  `ContainerScope`, `OnStop`, `SandboxRunConfig`, `GatewayInvokeRequest`, `GatewayResponse`,
  `SandboxEndpoint`.
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
199008 launcher, 199009 provider, 199010 isolation key, 199011 file version conflict.
Message builders (`fs_execution_message`, `shell_execution_message`, `code_execution_message`,
…) reproduce the Python templates character for character.

### Types

`fs::types` — `ReadFileParams/Data`, `ReadChunkData`, `WriteFileParams/Data`,
`UploadFileParams/Data`, `DownloadFileParams/Data`, `TransferChunkData`, `ListFilesParams`,
`ListDirsParams`, `ListFilesData`, `FileSystemItem` (Python `type` field), `SearchFilesParams/Data`,
`SearchPatternMode`, `FileStat`, `MkdirParams/Data`, `DeleteParams/Data`, `CopyParams/Data`,
`MoveParams/Data`, `ReadMode`, `FileContent`, `LineRange`, `SortBy`.
`shell::types` — `ShellType`, `StreamKind`, `ShellExecParams`, `ShellArgvParams`, `ShellOutput`,
`ShellChunk`, `ShellBackgroundOutput`.
`code::types` — `Language`, `CodeExecParams`, `CodeOutput`, `CodeChunk`.
All parameter structs are `serde`-deserializable with Python defaults and derive `JsonSchema`,
so [`tool`] builds tool schemas and tool-call argument parsing from the same definition.

### Support modules

`workspace::PathPolicy` (resolution + traversal protection), `encoding` (detection, BOM,
binary sniffing, decode/encode), `env` (`ExecutionEnvironment`, `resolve_environment`),
`util` (timestamps, octal permissions, SHA-256, UTF-8-safe truncation), `tool`
(`OperationToolSpec` generation), `constants`.

## Constraints

- Errors: every backend call returns `OpResult<T>`; constructors return
  `openjiuwen_common::error::Result`. Failures never panic and never raise.
- Async-first: tokio I/O only, no blocking calls on an async task (heavy CPU work uses
  `spawn_blocking`); streaming is `BoxStream<'static, OpResult<T>>`.
- Public types are `serde`-serializable, tolerate unknown fields, and default missing
  collections; cards round-trip through JSON.
- `tracing` logging only, no `println!`; no `unsafe`; files ≤1000 lines.
- Sandbox configuration types always compile; the HTTP client is behind the `sandbox` feature.
- Tests are offline and deterministic; they must not require `python3`/`node` to exist
  (skip with a message when the runtime is missing) and must not touch the network.

## Python parity map

| Python (`core/sys_operation/…`) | Rust |
|---|---|
| `sys_operation.py` (`SysOperation`, `SysOperationCard`, `generate_isolation_key_template`) | `sys_operation.rs`, `card.rs`, `sandbox/config.rs` |
| `base.py` (`OperationMode`, `BaseOperation`) | `mode.rs`, `operation.rs` |
| `config.py` (`LocalWorkConfig`, sandbox models) | `config.rs`, `sandbox/config.rs` |
| `fs.py`, `shell.py`, `code.py` (ABCs) | `operation.rs` traits |
| `result/{fs,shell,code}_operation_result.py`, `result/base_result.py` | `fs::types`, `shell::types`, `code::types`, `result.rs` |
| `local/fs_operation.py` (1755 lines) | `fs/local.rs`, `fs/slice.rs`, `fs/list.rs`, `fs/search.rs`, `fs/ext.rs` |
| `local/shell_operation.py` (1049 lines) | `shell/local.rs`, `shell/allowlist.rs`, `shell/plan.rs` |
| `local/code_operation.py` | `code/local.rs` |
| `local/utils.py` (`OperationUtils`, streaming decoding) | `env.rs`, `shell/local.rs`, `code/local.rs` |
| `local/_text_encoding.py`, `_rw_lock_manager.py` | `encoding.rs`, `fs/local.rs` (in-process per-path locks) |
| `cwd.py` (`get_cwd`/`get_workspace`) | `workspace::PathPolicy` |
| `execution_environment.py` | `env.rs` |
| `shell_process_registry.py` | `shell/registry.rs` |
| `tool_adapter.py` | `tool.rs` |
| `protocal/*.py`, `sandbox/sandbox_registry.py` | `sandbox::provider` |
| `sandbox/{fs,shell,code}_operation.py`, `sandbox/gateway/*` | `sandbox::operation` |
| `sandbox/launchers/*`, `providers/*`, `extensions/sys_operation` | deferred (external service) |

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
- `encoding="auto"` detects the encoding; BOM is stripped.
- Read failures are `OpResult` values (code 199003, `execution` = method name), never panics.

Write:
- `prepend_newline` defaults to **true** (a plain write gets a leading `\n`); `append_newline`
  defaults to false; `append` does not insert a newline and creates a missing file.
- `create_if_not_exist = false` on a missing file → 199003. `append` + missing file creates it.
- `expected_content_sha256` mismatch → `199011` with message `"file version conflict"` and the
  file left byte-identical; `create_if_absent` distinguishes an empty file from a missing one
  and rejects both when the file exists.
- `permissions` is applied best-effort on Unix as an octal string; not on Windows.

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

Shell (`local/test_shell_operation.py`):
- Empty/blank command → 199004 `"command can not be empty"`, `exit_code = -1`.
- Timeout: `timeout_secs` `None`/`0` → configured default (300). Expiry kills the child,
  keeps partial output, sets `timed_out`, `exit_code = -1`, and returns 199004 containing
  `"timeout"`.
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
  `"process exited early with code <n>: <stderr>"` (stderr retained up to 8192 bytes).

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
  `default_timeout_secs` bounds runtime; `max_depth` bounds traversal. Processes are killed on
  timeout and on cancellation through `shell::registry`.
- **Environment**: `EnvPolicy` can switch from Python's inherit-everything default to a
  minimal allowlist and can scrub secrets; explicit values can never re-add a scrubbed name.
- **No `unsafe`**, no panics on malformed paths/commands, no `unwrap` outside tests.
- Sequential-consistency for concurrent writes is provided by per-path in-process locks
  (Python uses on-disk SQLite locks shared across processes; a single process owns a
  workspace here).

## Known Issues / Deferred

- Remote sandbox protocol: the Python package speaks no HTTP in `core/`; its providers live in
  `extensions/sys_operation/sandbox/providers` and talk to an external AIO container service
  whose wire contract is unspecified here. `sandbox::http` (feature `sandbox`) carries the
  request/response envelope, registers forwarding backends through `install`, and exposes
  `HttpSandboxGateway::probe` as the **only** network call (a `GET` on the endpoint
  `base_url`); `invoke` — and therefore every sandbox operation — returns `Error::Unsupported`
  and is surfaced as `OpResult` 199007 with `sandbox_gateway_message("<method>", <reason>)`.
  Still deferred: launcher lifecycle (create/pause/resume/delete), endpoint discovery and
  `sandbox_id` assignment, isolation-key → container mapping, provider-specific payload
  schemas, the streaming wire format (streams call `"<method>_stream"` and expect a JSON chunk
  array, so in practice they yield one error chunk), and auth-header/auth-query injection.
- Legacy CJK encodings (GBK, Big5, Shift-JIS) are not decodable; `encoding` reports
  `Error::Unsupported` instead of producing mojibake. UTF-8/UTF-16/Latin-1 are supported.
- Writes are always UTF-8 (or UTF-16 when explicitly requested); Python can write arbitrary
  codecs.
- Cross-process file locks (Python's SQLite lock manager) are not ported; only in-process locks
  are provided.
- POSIX streaming does not apply Python's `stdbuf`/`script` buffering wrappers: output arrives
  in pipe-sized blocks and never through a PTY.
- Background early-exit detection uses a 100 ms grace window (Python defaults to 3 s and
  exposes `grace=`), so a command failing later than that is reported as started, with a pid.
- Cancellation kills the direct child only (`start_kill`): Python's `killpg` process-group kill
  and `kill_session_tree`/`consume_cancelled` are not ported (`cancel_all` covers teardown).
- `execute_argv` applies the allowlist to `argv[0]` but not `dangerous_patterns`: arguments are
  data, never shell text.
- Windows `Auto` uses `cmd /C`; Python's POSIX/PowerShell command classification heuristics and
  `-lc` path normalization are not ported.
- Shell and code output capture is bounded by `max_output_bytes` (Python captures unbounded);
  timeout and cancellation failures keep partial output in `data`.

## Test Strategy

- `#[cfg(test)]` modules next to the code cover unit behavior: parameter defaults, slicing,
  encoding detection, allowlist parsing, path containment, truncation, result mapping.
- Shell and code behavior ports live in `#[cfg(test)]` modules beside the backends:
  allowlist acceptance/rejection, deny patterns, timeout with partial output, bounded capture,
  env overlay/scrubbing, `execute_argv` without a shell, background early exit, stream chunk
  shape, session/instance cancellation, code language metadata and temp-file strategy.
- `tests/` holds cross-module behavior ports, each citing the Python test it ports:
  `card_json.rs` (card/config JSON parity), `local_fs.rs` (read/write/list/search/upload/
  download against `tempfile` dirs), `fs_edges.rs` (conditional write, binary policy, max size,
  line ranges), `traversal.rs` (escape attempts incl. symlinks), `shell_local.rs` (allow/deny,
  timeout, truncation, env overlay, argv form), `code_local.rs` (Python/JavaScript, skipped when
  the runtime is absent), `tool_specs.rs` (13 specs, ids, schemas), `sandbox_stub.rs`
  (feature `sandbox`: unsupported remote calls, isolation keys, provider dispatch).
- `src/sandbox/http.rs` and `src/sandbox/operation.rs` keep their tests inline: fake-provider
  forwarding (exact `OpResult` and stream passthrough), dispatch + per-session isolation-key
  resolution, missing provider (`ErrorKind::Unsupported`) and missing gateway/launcher config
  (199002) paths, `install` idempotency, and one `#[ignore]`d live `probe` test.
- Every test is offline and deterministic: `tempfile` workspaces, no network, and runtime
  lookups (`python3`, `node`, `bash`) skip with an explanatory message when missing.
- Gates: `cargo fmt -p openjiuwen-sysop`, `cargo test -p openjiuwen-sysop`,
  `cargo test -p openjiuwen-sysop --features sandbox`,
  `cargo clippy -p openjiuwen-sysop --all-targets -- -D warnings`.

## Routing Table

- `./src/sys_operation.rs` → `SysOperation` facade, mode dispatch, tool spec aggregation
- `./src/card.rs` → `SysOperationCard`, validation, tool-id generation
- `./src/config.rs` → `LocalWorkConfig`, `EnvPolicy`, `BinaryPolicy`, `DEFAULT_DANGEROUS_PATTERNS`
- `./src/mode.rs` → `OperationMode`, `OperationKind`
- `./src/status.rs` → `SysOpStatus`, Python message templates
- `./src/result.rs` → `OpResult`, failure builders, `Result` conversion
- `./src/operation.rs` → `FsOperation`, `FsExtOperation`, `ShellOperation`, `CodeOperation`, `OpStream`, `OperationSet`
- `./src/workspace.rs` → `PathPolicy`, path cleaning, symlink resolution, containment checks
- `./src/encoding.rs` → encoding detection, BOM handling, binary sniffing, decode/encode
- `./src/env.rs` → `ExecutionEnvironment`, `resolve_environment`, env-name validation
- `./src/util.rs` → timestamp formatting, octal permissions, SHA-256, UTF-8-safe truncation
- `./src/tool.rs` → `OperationToolSpec` generation from parameter schemas
- `./src/constants.rs` → defaults shared by the backends
- `./src/fs/` → filesystem types and the local backend (`local`, `slice`, `list`, `search`, `ext`)
- `./src/shell/types.rs` → `ShellType`, `ShellExecParams`, `ShellArgvParams`, `ShellOutput`, `ShellChunk`, `ShellBackgroundOutput`
- `./src/shell/allowlist.rs` → `CommandFilter`: first-token allowlist (quote-aware splitter) and dangerous-pattern deny list
- `./src/shell/plan.rs` → `resolve_plan` per `ShellType` and OS, absolute-path literal extraction
- `./src/shell/registry.rs` → `ShellProcessRegistry`: per-session cancellation tokens, RAII registration, `global()`
- `./src/shell/local.rs` → `LocalShellOperation` and the shared bounded-capture/timeout engine (`pub(crate) mod process`)
- `./src/code/types.rs` → `Language`, `CodeExecParams`, `CodeOutput`, `CodeChunk`
- `./src/code/local.rs` → `LocalCodeOperation`: runtime lookup, inline/temp-file strategy, forced env
- `./src/sandbox/` → gateway config, provider registry, sandbox dispatch + forwarding backends, HTTP gateway client (feature `sandbox`)
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
- `sandbox::operation` resolves everything in a private core that takes a
  `&SandboxProviderRegistry`: tests exercise dispatch against a local registry and never
  register mocks in `sandbox::provider::global()` (only `SysOperation` uses the global one).
- `sandbox::operation::register_builtin_providers` is idempotent: it is inert without the
  `sandbox` feature or without a launcher `base_url`, and duplicate registrations are treated
  as success, so repeated `SysOperation::new` calls never fail.
- Sandbox forwarding backends (`SandboxFsOperation`, `SandboxShellOperation`,
  `SandboxCodeOperation`) never serialize, inspect, or re-map provider results: the provider
  owns the wire format. `supports_conditional_write` returns the provider's answer; the card's
  claim is exposed separately as `configured_conditional_write`.
- Streams must be `'static`; use `self: Arc<Self>` receivers and `async_stream::try_stream!`,
  yielding a single error chunk on failure instead of returning `Err`.
- The local shell and code backends share `crate::shell::local::process` (`spawn_child`,
  `apply_environment`, `run_captured`, `stream_child`); never re-implement bounded capture,
  timeout, or cancellation inside `code/local.rs`.
- Shell gate order is blank command → deny list → allowlist → cwd/sandbox → shell plan, so a
  rejected command never spawns a child; `CommandFilter::check` reports the deny list first.
- `<lang> is not supported` is unrepresentable here (`Language` is a closed enum), and
  `CodeExecParams::encoding` is accepted but ignored: decoding is UTF-8 lossy.
- `src/shell/local.rs` carries the facade plus the shared engine (~1000 lines of code plus
  ~350 lines of tests); if it grows further, extract `process` into `src/shell/process.rs`.
