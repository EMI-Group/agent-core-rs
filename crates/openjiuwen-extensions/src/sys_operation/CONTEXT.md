# sys_operation/ — sandbox providers and registration facade

## Intent

Python's `openjiuwen.extensions.sys_operation.sandbox.providers` registers three sandbox providers (`aio`, `jiuwenbox`, `yuanrong`) into the kernel sandbox registry.
This crate implements those three providers for real, as HTTP clients in `remote/`, and keeps the registration facade in `mod.rs` that hands the kernel's `SandboxProviderRegistry` complete factory sets.
Pick this node for "register a sandbox provider", "where is AIO/JiuwenBox/YuanRong", or "which HTTP endpoints does a sandbox provider speak".
Transport, wire mapping, and host-side operation semantics live in `./remote/`; `mod.rs` only builds and registers factories.

## API Surface

The seam is defined in `openjiuwen-sysop` (`sandbox/provider.rs`, `sandbox/config.rs`) and reaches this crate through the kernel umbrella path `openjiuwen_core::sys_operation::sandbox::{provider, config}` — the path is re-exported unchanged.

```rust
pub mod remote;
pub use openjiuwen_core::sys_operation::sandbox::config::{SandboxEndpoint, SandboxGatewayConfig};

/// The sandbox types with a real HTTP transport: `["aio", "jiuwenbox", "yuanrong"]`.
pub const REMOTE_SANDBOX_TYPES: [&str; 3] = ["aio", "jiuwenbox", "yuanrong"];
/// Sandbox types Python registers but this build cannot provide: none.
pub const DEFERRED_SANDBOX_TYPES: [&str; 0] = [];

/// The three factories Python registers per sandbox type; any subset may be present.
pub struct SandboxProviderSet { pub fs: Option<FsProviderFactory>, pub shell: Option<ShellProviderFactory>, pub code: Option<CodeProviderFactory> }
impl SandboxProviderSet {
    pub fn new() -> Self;
    pub fn is_empty(&self) -> bool;
    pub fn with_fs(self, factory: FsProviderFactory) -> Self;
    pub fn with_shell(self, factory: ShellProviderFactory) -> Self;
    pub fn with_code(self, factory: CodeProviderFactory) -> Self;
    /// A set whose three factories answer `Error::Unsupported` naming `sandbox_type`.
    pub fn deferred(sandbox_type: impl Into<String>) -> Self;
}

/// The factory set of one remote sandbox type (aio, jiuwenbox, yuanrong).
pub fn remote_provider_set(sandbox_type: &str) -> Result<SandboxProviderSet>;
/// Registers the remote provider set of one sandbox type.
pub fn register_remote(registry: &SandboxProviderRegistry, sandbox_type: &str) -> Result<()>;
/// Registers every type in `REMOTE_SANDBOX_TYPES`, in declaration order.
pub fn register_remote_providers(registry: &SandboxProviderRegistry) -> Result<()>;

/// Registers every factory present in `providers` under `sandbox_type`.
pub fn register(registry: &SandboxProviderRegistry, sandbox_type: &str, providers: SandboxProviderSet) -> Result<()>;
/// Compatibility no-op: registers nothing, logs at debug, never fails.
pub async fn register_deferred(registry: &SandboxProviderRegistry) -> Result<()>;

/// The sandbox types with at least one registered provider, sorted.
pub fn registered_types(registry: &SandboxProviderRegistry) -> Vec<String>;
/// Whether the sandbox type is deferred: `true` for any name outside `REMOTE_SANDBOX_TYPES`.
pub fn is_deferred(sandbox_type: &str) -> bool;

/// Registers a provider set bound to a shared `SandboxGatewayConfig`.
pub fn register_gateway_provider(registry: &SandboxProviderRegistry, sandbox_type: &str, gateway: SandboxGatewayConfig, providers: SandboxProviderSet) -> Result<()>;
```

`remote_provider_set` and `register_remote` reject an unknown type with `ErrorKind::Unsupported` and the message `sandbox provider '<type>' is not a remote provider; supported types: aio, jiuwenbox, yuanrong` — never a silent success.
`register_remote_providers` installs `aio`, then `jiuwenbox`, then `yuanrong`, stopping at the first failure (a duplicate is the kernel's `ErrorKind::SysOp`).

## Registration Semantics

- Registration is keyed by `(sandbox_type, operation)` exactly like Python's `SandboxRegistry.provider(sandbox_type, operation_type)`; duplicate registration is the kernel's `Error::SysOp`, not replaced silently.
- Registration is synchronous, offline, and cheap: the factories capture configuration, and the transport contacts the container only when an operation runs.
- Registration is not transactional: a set that fails part-way (duplicate or kernel error) leaves the factories registered before the failure in place.
- `register_deferred` stays `async` to mirror the Python async setup path, performs no I/O, and exists only so old call sites keep compiling; the Python sandbox types are real providers now, so registration goes through `register_remote_providers` / `register_remote`.
- `is_deferred(t)` is `!REMOTE_SANDBOX_TYPES.contains(&t)`, so `is_deferred("aio") == false` and any unknown name is deferred.
- `SandboxProviderSet::deferred(sandbox_type)` is kept for host-defined sandbox types that are configured but not implemented: its factories return `Error::Unsupported("sandbox provider '<type>' is deferred; register a provider with sys_operation::register")`.
- `register_gateway_provider` exists so a host can plug its own transport (or a mock in tests) without this crate shipping a protocol; it binds the supplied `SandboxGatewayConfig` into every registered factory, so the configuration passed to `create_fs`/`create_shell`/`create_code` is ignored for those registrations.

## Provider Transports

| Type | Base URL | Auth | Container create | Wire endpoints |
|---|---|---|---|---|
| `aio` | `endpoint.base_url` (required) + `invoke_path` | card `auth_headers` / `auth_query_params` | none (an AIO host already owns the sandbox) | `POST {base_url}{invoke_path}` carrying the kernel gateway envelope |
| `jiuwenbox` | `endpoint.base_url`, falling back to `launcher_config.base_url`; bare `host:port` gains `http://` | card `auth_headers` / `auth_query_params`, plus `Authorization: Bearer $JIUWENBOX_API_TOKEN` when set (an explicit `Authorization` header wins) | `POST /api/v1/sandboxes` | `PUT /api/v1/timeout`; `GET`/`DELETE /api/v1/sandboxes/{id}`; `POST …/exec`; `POST …/upload`; `GET …/download`; `GET …/files`; `GET …/search` |
| `yuanrong` | `endpoint.base_url` (required), falling back to `launcher_config.base_url` | card `auth_headers` / `auth_query_params`; `Accept: application/json` on ADX calls | `POST /api/environment/v1/environments` | `POST /direct/{urlencoded id}/invoke`; `DELETE /api/environment/v1/environments/{id}` |

The `aio` envelope is `GatewayInvokeRequest { op_type: "fs"|"shell"|"code", method, params, isolation_key }` answered by `GatewayResponse { code, message, data }`; see `./remote/aio/` for why that contract and not the third-party SDK's REST surface is the transport.
JiuwenBox's `exec` body is `{ command: [argv…], workdir?, env?, stdin?, timeout_seconds? }` returning `{ exit_code, stdout, stderr }` (exit `124` means timeout); upload is multipart field `file` with `?sandbox_path=`, download returns raw bytes.
YuanRong's create body is `{ rootfs (JSON string), cpu: 1000, memory: 1024, idleTimeoutSeconds, runtime: supervisor|docker|conch, extra_config? }`; actions are `process.exec {cmd, cwd?, timeout?}`, `file.read {path, binary}`, `file.write {path, data, binary, append}`, `file.list {path, depth}` (depth `1` = non-recursive), and `file.search {path, pattern?, exclude_patterns?}`.
All three providers are always compiled — no Cargo feature gates them — and the default build never contacts a container.

## Lifecycle and Caching

- The kernel has no launcher or lifecycle API: it resolves a `SandboxEndpoint { base_url, sandbox_id, isolation_key }` itself and calls the factories, so each provider creates its container lazily.
- Containers are cached in a process-global `OnceLock<tokio::sync::Mutex<BTreeMap<String, Container>>>` keyed by `"{base_url}|{isolation_key}"` (an absent isolation key contributes the empty string), which is Python's `_shared_sandbox_ids` / `_shared_instances`; the fs, shell, and code handles of one scope therefore share one container.
- JiuwenBox resolves its sandbox id in order: `JIUWENBOX_SANDBOX_ID`, `launcher.extra_params["sandbox_id"]`, the endpoint's `sandbox_id`, the shared cache, then `POST /api/v1/sandboxes`; resolution runs under a per-backend lock so concurrent first calls create once.
- JiuwenBox writes the server idle policy once per base URL through `PUT /api/v1/timeout` (`launcher.idle_ttl_seconds` + `extra_params["idle_check_interval"]`), skipped when neither is configured and logged at `debug` on failure.
- YuanRong resolves its `(environment_id, runtime)` in order: the endpoint's `sandbox_id`, the shared cache, then a create.
- A lost container is recreated with bounded retries: JiuwenBox treats a `404` whose body reads `Sandbox '<id>' not found` as gone (`JIUWENBOX_SANDBOX_RECREATE_RETRIES`, default `3`, `0` disables), deletes the stale container best-effort, waits one second, and recreates; YuanRong treats an invoke `404` the same way with `min(retries + 1, 6)` attempts (default five retries).
- Factories never fail for a missing container: the first operation establishes or authenticates, so building a provider is pure.

## Error Mapping

- Any transport, HTTP-status, or gateway-envelope failure becomes the kernel's `SysOpStatus::SandboxGatewayError` (`199007`) through `sandbox_gateway_message(operation, reason)` — for filesystem, shell, and code operations alike, on every provider.
- Operation-specific codes belong to validation and outcomes only: blank command `199004 command can not be empty`, blank code `199005 code can not be empty`, shell timeout `199004 execution timeout after <n> seconds` (with partial output in `data`), code timeout `199005` likewise, and unsupported filesystem requests `199003` from the `fs_failure` helpers.
- Auth header and query values, tokens, and file contents are never logged; failures log the operation name and a redacted reason at `debug`.
- `Error::config(..)` is returned only at construction (for example `yuanrong provider requires endpoint.base_url`) through `openjiuwen_common::error::Error`.
- A provider never reports success for a failed backend: a non-zero exit code stays a successful *execution* (`OpResult` with `exit_code`), while an unreachable or refusing gateway is always a failure frame.

## Kernel Seam Facts

- `SandboxProviderRegistry` is public, `Clone + Default + Send + Sync`, and built with `new()`; clones share the same `Arc<RwLock<BTreeMap<..>>>` maps.
- Registration is per `(sandbox_type, operation)`: `register_fs` / `register_shell` / `register_code` take `impl Into<String>` and return the kernel's `Error::SysOp` on a duplicate.
- `create_fs` / `create_shell` / `create_code(&self, sandbox_type, &SandboxEndpoint, &SandboxGatewayConfig)` return `Result<Arc<dyn …Operation>>` and answer `Error::Unsupported("Sandbox type '<t>' does not support operation '<op>'")` for an empty slot.
- Factories are synchronous `Arc<dyn Fn(&SandboxEndpoint, &SandboxGatewayConfig) -> Result<Arc<dyn …>> + Send + Sync>`, so any handshake must be deferred into the returned backend (this crate does that with locks and `OnceCell`-style resolution).
- `sandbox_types()` enumerates only the filesystem map, so a shell/code-only registration is invisible to `registered_types`.
- `provider::global()` is the process-global registry, and the kernel dispatch path (`SysOperation` in sandbox mode) resolves **only** from it; registering on a local registry never reaches a `SysOperation`.
- The endpoint's `sandbox_id` is `None` for a kernel-built remote endpoint, so container identity must live in the provider keyed by `isolation_key`.
- `SandboxLauncherConfig.on_stop`, `idle_ttl_seconds`, `gateway_url`, and `extra_params` are pure configuration data; a provider must interpret them itself, and no provider is ever called on release.
- `GatewayInvokeRequest` / `GatewayResponse` live in the always-compiled `sandbox::config`; `sandbox::http::HttpSandboxGateway` sits behind sysop's opt-in `sandbox` feature, its `invoke` returns `Error::Unsupported`, and only `probe` performs a GET — this crate uses `reqwest` directly instead.
- `openjiuwen-harness` only ever builds `SysOperation::local(..)` and never touches the registry, so no harness code path exercises a remote provider today.

## Known Issues and Gaps

- No teardown on release: the kernel never calls a provider when a handle drops, so containers outlive the process unless the deployment's idle reaper collects them; `on_stop`, `idle_ttl_seconds` (beyond JiuwenBox's idle policy), and gateway idle eviction are not applied at release.
- Streaming methods are buffered: no transport offers a streaming endpoint, so each stream performs its buffered call and yields host-side chunks (text by line, bytes by `chunk_size`, shell/code output by line) ending in a terminal `is_last_chunk` frame — the same shape Python produces.
- Transfers hold the whole file in host memory: `upload_file`/`download_file` and their stream variants read the local file or the remote bytes at once before chunking.
- Conditional writes are unsupported by the remote transports: `write_file` rejects `expected_content_sha256` and `create_if_absent` with a clear `fs_failure`, and `supports_conditional_write()` reflects `SandboxGatewayConfig.supports_conditional_write` (AIO) rather than a backend capability.
- `execute_argv` has no Python counterpart; JiuwenBox and YuanRong send argv as a list or a POSIX-quoted command, and AIO sends the POSIX-quoted form through `execute_cmd`.
- `execute_cmd_background` runs the command detached inside the container and returns a handle-only result where the transport allows it; there is no streamed or interactive background channel, and Windows host quoting is not attempted.
- JiuwenBox's hybrid local/remote shell routing (`plan_command_rewrite`, tree-sitter classification, `exclude_commands`), the host fallback of `_run_exec_pipeline`, lifecycle hooks, `preserve_files_upload`, Unix-domain-socket transport (`unix://…`, `JIUWENBOX_LISTEN`), and the caller-facing `force_recreate`/`delete` helpers are not ported.
- YuanRong always writes binary hex (`file.write {binary: true}`) and reads a binary payload (base64 when `content_encoding` is `base64`, hex otherwise); Python's plain-text path is not used.
- YuanRong `file_types` filtering is host-side exact matching on the ADX `type` field, and unknown ADX types pass through unchanged.
- AIO talks the kernel gateway envelope because the third-party `agent_sandbox` SDK and its private REST contract are not part of the reference repository; the envelope is the only gateway contract the reference defines, and `params` field names follow the Python provider method signatures (`timeout`, `line_range` as `[start, end]`, padded base64 content with `encoding: "base64"`).
- AIO `upload_file` cannot hand a host `local_path` to the container, so it reads the local file and inlines it as base64 in a `write_file` envelope call.

## Constraints

- Async-first: tokio I/O with `reqwest`, no blocking calls on async tasks, public streams are `BoxStream<'static, OpResult<T>>`.
- Errors: `openjiuwen_common::error::{Error, Result}` for construction and validation; operation failures are `OpResult` values built with the kernel's `fs_failure` / `shell_failure` / `code_failure` / gateway-status helpers.
- Logging: `tracing` only, target `openjiuwen_extensions::sys_operation`; never log auth values, tokens, or payload contents above `debug`.
- Safety: no `unsafe`, no panics on user input, no global mutable state beyond the documented container caches.
- Registration stays synchronous; idempotence is the caller's problem (the kernel rejects duplicates).
- Files ≤ ~1000 lines, split by concern; `remote/backend/tests.rs` and `remote/aio/tests/fs.rs` are exhaustive operation matrices kept whole on purpose.

## Test Strategy

- `mod.rs` unit tests cover the facade: empty `DEFERRED_SANDBOX_TYPES`, the three `REMOTE_SANDBOX_TYPES`, a complete factory set per remote type, unknown-type rejection messages, `is_deferred` semantics, duplicate registration, `register_gateway_provider` binding, and the `register_deferred` no-op.
- `remote/transport.rs` tests cover base-URL joining, `auth_headers`/`auth_query_params` injection, bearer tokens, timeouts, status rendering, hex/base64 round-trips, and `gateway_failure` producing `199007` with the Python message.
- `remote/semantics.rs` and `remote/local_io.rs` test the pure host-side helpers: line selection and its mutual-exclusion errors, newline transforms, chunking, POSIX quoting, sorting/filtering, and host reads/writes with `overwrite`/`create_parent_dirs`.
- `remote/backend/tests.rs` drives the generic wrappers through a recording fake backend: every `FsOperation`/`ShellOperation`/`CodeOperation` method, stream termination, `supports_conditional_write` propagation, and error-code mapping.
- `remote/aio/tests.rs` plus `aio/tests/{fs,shell,code}.rs` use `wiremock` on `127.0.0.1:0` to assert envelope bodies, `invoke_path` overrides, auth injection, `code != 0` → `199007`, transport failures, conditional-write flags, and stream termination.
- `remote/jiuwenbox/tests.rs` and `remote/yuanrong/tests.rs` assert request paths, query parameters, create bodies, argv/exec encoding, id caching and recreate-on-404, idle-timeout dedupe, ADX action payloads, and error mapping.
- No live network is used: every HTTP interaction targets a loopback `wiremock` server and each test uses a distinct isolation key so the process-global container cache never leaks between tests.
- Python counterparts are named in comments (`tests/unit_tests/extensions/sys_operation/sandbox/*.py`).

## Routing Table

- `./mod.rs` → `SandboxProviderSet`, registration helpers, remote/deferred type lists, `from`-scratch facade tests
- `./remote/` → the three real providers, shared transport, host-side semantics, and their suites
- `./remote/transport.rs` → `SandboxHttp` (base URL, timeout, auth injection), `RemoteError`/`RawResponse`, `gateway_failure`, hex/base64 codecs
- `./remote/semantics.rs` → line selection, newline transforms, chunking, path/quoting helpers, list sorting and `file_types` filtering
- `./remote/local_io.rs` → host-side file reads and writes for transfers
- `./remote/backend/` → `SandboxBackend` trait, request/result types, `provider_set_from`, and the generic `RemoteFs`/`RemoteShell`/`RemoteCode` wrappers (`fs.rs`, `shell.rs`, `code.rs`, `tests.rs`)
- `./remote/aio/` → AIO gateway-envelope client and its `fs`/`shell`/`code` wrappers, tests in `tests.rs` + `tests/`
- `./remote/jiuwenbox.rs` → JiuwenBox REST backend (container create/cache, exec, upload/download, files, search) + `./remote/jiuwenbox/tests.rs`
- `./remote/yuanrong.rs` → YuanRong ADX backend (environment create/cache, `process.exec`, `file.*`) + `./remote/yuanrong/tests.rs`
