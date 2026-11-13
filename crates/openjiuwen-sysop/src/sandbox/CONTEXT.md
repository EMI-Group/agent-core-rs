# src/sandbox

## Intent

The sandbox backend seam: a sandbox-mode `SysOperationCard` names a `sandbox_type`, and a provider registered under that key builds the same filesystem/shell/code backends the local mode implements.
`openjiuwen-extensions::sys_operation` is the intended out-of-crate registrar; this module only supplies the registry, the configuration shape, and dispatch.

## API Surface

- `mod.rs`: `pub mod config; pub mod operation; pub mod provider;`, plus `#[cfg(feature = "sandbox")] pub mod http;` — the config/provider/operation types are always compiled, only the HTTP client is feature-gated.
- `provider` — `FsProviderFactory`/`ShellProviderFactory`/`CodeProviderFactory` are `Arc<dyn Fn(&SandboxEndpoint, &SandboxGatewayConfig) -> Result<Arc<dyn ...Operation>> + Send + Sync>`; `SandboxProviderRegistry` (`new`, `register_fs`/`register_shell`/`register_code`, `unregister`, `has_*`, `sandbox_types`, `create_fs`/`create_shell`/`create_code`) is `Clone + Default` with shared `Arc<RwLock<BTreeMap<..>>>` maps, keyed by `sandbox_type` only.
- `provider::global()` — `&'static SandboxProviderRegistry` process-global singleton; `sandbox_operations*` resolve only through it.
- `operation` — `sandbox_operations(&card)`, `sandbox_operations_for_session(&card, Option<&str>)`, `register_builtin_providers(&registry, &gateway)`, `unsupported_sandbox(&str)`, and the forwarding wrappers `SandboxFsOperation`/`SandboxShellOperation`/`SandboxCodeOperation` (each exposes `endpoint()` and `isolation_key()`; `SandboxFsOperation` also `configured_conditional_write()`).
- `config` — `ContainerScope`, `OnStop`, `SandboxIsolationConfig`, `SandboxLauncherConfig`, `SandboxGatewayConfig`, `SandboxRunConfig`, `GatewayInvokeRequest`, `GatewayResponse`, `SandboxEndpoint`, `SESSION_ID_PLACEHOLDER`, `generate_isolation_key_template`, `resolve_isolation_key`, `isolation_key_template_from_config`.
- `http` (feature `sandbox`) — `HttpSandboxGateway { new, endpoint, config, invoke, probe }`, `HttpFsOperation`/`HttpShellOperation`/`HttpCodeOperation` (`new(gateway: Arc<HttpSandboxGateway>)`, `gateway()`), `install(&registry, sandbox_type, &endpoint, &config)`.

## Constraints

- Provider factories are synchronous `Fn`, not async: a remote protocol handshake (endpoint discovery, auth token, container create) must happen lazily inside the backend on first operation or block inside the factory.
- Dispatch builds `SandboxEndpoint` itself: `base_url = launcher.base_url.unwrap_or_default()`, `sandbox_id = None`, `isolation_key = resolve_isolation_key(template, session_id)`; providers cannot influence the endpoint, only consume it.
- `register_builtin_providers` is inert unless the `sandbox` feature is on and `launcher.base_url` is non-blank; `install` checks `has_fs`/`has_shell`/`has_code` first, so a provider registered externally before construction wins, while a later duplicate `register_*` returns `ErrorKind::SysOp` ("already registered").
- Dispatch is eager: a sandbox card without a resolvable provider fails at `SysOperation::new`, not on first call; providers are recreated per `SysOperation`.
- No create/pause/resume/delete lifecycle types exist: only the `OnStop::{Delete,Pause,Keep}` enum and `SandboxLauncherConfig` fields (`on_stop`, `idle_ttl_seconds`, `gateway_url`, `extra_params`, `base_url`); nothing assigns `sandbox_id` or maps an isolation key to a container.
- `SandboxGatewayConfig::auth_headers`/`auth_query_params` are carried but never applied; `GatewayInvokeRequest`/`GatewayResponse` are the only wire-envelope types.
- `sandbox_types()` enumerates the filesystem map only, so a type registered with shell/code alone is not listed.
- `openjiuwen-core` (and therefore `openjiuwen-extensions`) does not forward the `sandbox` feature; the `http` module compiles only for a direct `openjiuwen-sysop` dependency with `features = ["sandbox"]`.
- Forwarding wrappers never serialize, inspect, or re-map parameters or `OpResult`; failures stay values with code 199007 and `"sandbox gateway error, operation: <method>, reason: <reason>"`.

## Routing Table

- `./mod.rs` → module declarations and the `sandbox` feature gate
- `./provider.rs` → factory type aliases, `SandboxProviderRegistry`, `global()`
- `./config.rs` → isolation/launcher/gateway/endpoint types and isolation-key helpers
- `./operation.rs` → dispatch (`sandbox_operations*`, `register_builtin_providers`) and the three forwarding wrappers
- `./http.rs` → feature-gated `HttpSandboxGateway` (invoke = `Unsupported`, probe = the only network call), HTTP forwarding backends, `install`
