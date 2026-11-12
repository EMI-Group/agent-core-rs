# sys_operation/ — sandbox provider registration facade

## Intent

Python's `extensions/sys_operation/sandbox/providers/` registers three sandbox providers (`aio`, `jiuwenbox`, `yuanrong`) into the kernel `SandboxRegistry` at import time.
Rust keeps the registration seam and drops the protocol implementations: this module hands the kernel's `SandboxProviderRegistry` factory sets, registers documented deferred providers that fail with `Error::Unsupported`, and exposes what is registered.
Pick this node for "register a sandbox provider" or "where is AIO/JiuwenBox/YuanRong". No protocol implementation lives here.

## API Surface

The seam is defined in `openjiuwen-sysop` (`sandbox/provider.rs`, `sandbox/config.rs`) and reaches this crate through the kernel umbrella path `openjiuwen_core::sys_operation::sandbox::{provider, config}` — the path is re-exported unchanged.

```rust
pub use openjiuwen_core::sys_operation::sandbox::config::{SandboxEndpoint, SandboxGatewayConfig};
// The provider seam types (`FsProviderFactory`, `ShellProviderFactory`, `CodeProviderFactory`,
// and `SandboxProviderRegistry`) are named through the kernel path
// `openjiuwen_core::sys_operation::sandbox::provider`; the `SandboxProviderSet` fields use them.

/// The three factories Python registers per sandbox type; any subset may be present.
pub struct SandboxProviderSet {
    pub fs: Option<FsProviderFactory>,
    pub shell: Option<ShellProviderFactory>,
    pub code: Option<CodeProviderFactory>,
}
impl SandboxProviderSet {
    pub fn new() -> Self;
    pub fn is_empty(&self) -> bool;
    pub fn with_fs(self, factory: FsProviderFactory) -> Self;
    pub fn with_shell(self, factory: ShellProviderFactory) -> Self;
    pub fn with_code(self, factory: CodeProviderFactory) -> Self;
    /// A set whose three factories answer `Error::Unsupported` naming `sandbox_type`.
    pub fn deferred(sandbox_type: impl Into<String>) -> Self;
}

/// Registers every factory present in `providers` for `sandbox_type`.
/// Stops at the first failure (duplicate registration surfaces the kernel's `Error::SysOp`).
pub fn register(registry: &SandboxProviderRegistry, sandbox_type: &str, providers: SandboxProviderSet) -> Result<()>;

/// Registers the deferred Python providers (`aio`, `jiuwenbox`, `yuanrong`).
pub async fn register_deferred(registry: &SandboxProviderRegistry) -> Result<()>;
pub const DEFERRED_SANDBOX_TYPES: [&str; 3] = ["aio", "jiuwenbox", "yuanrong"];

/// The sandbox types with at least one registered provider, sorted.
/// Enumerates the kernel's `sandbox_types()`, which is keyed on the filesystem map:
/// a set registered with only shell/code factories is not listed.
pub fn registered_types(registry: &SandboxProviderRegistry) -> Vec<String>;

/// Whether the sandbox type is deferred (documented, not implemented) rather than available.
pub fn is_deferred(sandbox_type: &str) -> bool;

/// Registers an HTTP gateway provider set built from a shared `SandboxGatewayConfig`
/// for hosts that already implement the gateway protocol elsewhere; `None` fields stay unregistered.
pub fn register_gateway_provider(
    registry: &SandboxProviderRegistry,
    sandbox_type: &str,
    gateway: SandboxGatewayConfig,
    providers: SandboxProviderSet,
) -> Result<()>;
```

## Ported Behavior

- Registration is keyed by `(sandbox_type, operation)` exactly like Python's `SandboxRegistry.provider(sandbox_type, operation_type)`; duplicate registration is the kernel's `Error::SysOp`, not replaced silently.
- Deferred providers keep Python's names so configuration written for Python keeps working: `aio` (agent-sandbox SDK), `jiuwenbox` (HTTP/UDS gateway, `JIUWENBOX_*` env vars), `yuanrong` (ADX HTTP API) — every factory returns `Error::Unsupported("sandbox provider '<type>' is deferred; register a provider with sys_operation::register")`.
- `supports_conditional_write` is the deferred provider's `false` (Python providers all report `False`); no launcher (`pre_deploy`) is registered — launchers belong to the kernel's sandbox config.
- `register_gateway_provider` exists so a host can plug its own transport (or a mock in tests) without this crate shipping a protocol.
- `register_gateway_provider` binds the supplied `SandboxGatewayConfig` into every registered factory, so the configuration a caller passes to `create_fs`/`create_shell`/`create_code` is ignored for those registrations.
- `register_deferred` is `async` only for parity with the Python async setup path; it performs no I/O and is the Rust equivalent of importing the Python `providers` package (hosts call it explicitly; there is no import-time side effect).
- Registration is not transactional: a set that fails part-way (duplicate or kernel error) leaves the factories registered before the failure in place.

## Constraints

- No new protocol implementation, no `reqwest` calls here; only factory registration.
- Factories are `Arc<dyn Fn(&SandboxEndpoint, &SandboxGatewayConfig) -> Result<...> + Send + Sync>`, matching the kernel aliases exactly.
- Registration is synchronous and idempotence is the caller's problem (the kernel rejects duplicates).
- Errors: propagate the kernel's `Error::SysOp`; never panic on a poisoned registry.
- Files ≤ ~1000 lines.

## Test Strategy

- Unit tests live inside `mod.rs`: `SandboxProviderSet::deferred("aio")` registers three factory types whose `create_*` calls return `ErrorKind::Unsupported` naming `aio`; `register_deferred` makes `registered_types` contain all three and `is_deferred` true; duplicate registration → `ErrorKind::SysOp`; a mock provider set registers and `create_fs` returns the mock; `register_gateway_provider` binds its shared configuration.
- Python counterparts named in comments: `tests/unit_tests/extensions/sys_operation/sandbox/test_aio_*.py` (registration only; behavior is integration-gated in Python and not ported).

## Routing Table

- `./mod.rs` → provider sets, registration helpers, deferred provider list, tests
