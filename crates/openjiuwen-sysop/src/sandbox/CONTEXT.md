# src/sandbox

## Intent

The sandbox backend seam: a sandbox-mode `SysOperationCard` names a `sandbox_type`, and a provider registered under that key builds the same filesystem/shell/code backends the local mode implements.
`openjiuwen-extensions::sys_operation` is the intended out-of-crate registrar for the provider transports (`aio`, `jiuwenbox`, `yuanrong`); this module supplies the registries, the configuration shape, the launcher lifecycle, and dispatch.
Two endpoint sources are supported: a static, pre-deployed `launcher_config.base_url`, and a container started on demand by a registered `SandboxLauncher` (the "managed" path).

## API Surface

- `mod.rs`: `pub mod config; pub mod launcher; pub mod lifecycle; pub mod operation; pub mod provider;`, plus `#[cfg(feature = "sandbox")] pub mod http;` — configuration, launcher, lifecycle, provider, and dispatch types are always compiled, only the HTTP client is feature-gated.
- `provider` — `FsProviderFactory`/`ShellProviderFactory`/`CodeProviderFactory` are `Arc<dyn Fn(&SandboxEndpoint, &SandboxGatewayConfig) -> Result<Arc<dyn ...Operation>> + Send + Sync>`; `SandboxProviderRegistry` (`new`, `register_fs`/`register_shell`/`register_code`, `unregister`, `has_*`, `sandbox_types`, `create_fs`/`create_shell`/`create_code`) is `Clone + Default` with shared `Arc<RwLock<BTreeMap<..>>>` maps, keyed by `sandbox_type` only.
- `provider::global()` — `&'static SandboxProviderRegistry` process-global singleton; `sandbox_operations*` resolve only through it.
- `launcher` — `SandboxStatus::{Running,Paused,Stopped,Unknown}` (`as_str`, `is_running`, serde snake_case), `LaunchedSandbox { base_url, sandbox_id, host_port }` (`new`, `with_sandbox_id`), `#[async_trait] trait SandboxLauncher` (required `launch(config, timeout_seconds, isolation_key)`, defaulted `pause`/`resume`/`delete` no-ops and `check_status → Running`), `PreDeploymentLauncher` (`launcher_type == "pre_deploy"`), `LauncherRegistry` (`new`, `register`, `has`, `get`, `create`, `unregister`, `launcher_types`), `global()`, `register_builtin_launchers`, and the error builders `unknown_launcher_error`/`launcher_error`/`provider_error`/`isolation_key_error`/`gateway_error` plus `launcher_op_failure`/`launcher_op_result`.
- `lifecycle` — `SandboxRecord { isolation_key, sandbox_id, endpoint, status, launcher_type, created_epoch_secs, last_used_epoch_secs }`, `trait Clock` + `SystemClock` + `FixedClock`, `now_epoch_secs()`, and `SandboxLifecycle` (`new`, `with_clock`, `launchers`, `now`, `record`, `records`, `len`, `is_empty`, `ensure_endpoint(config, isolation_key, now)`, `ensure_endpoint_now`, `release(isolation_key, on_stop)`, `evict_idle(ttl, now)`, `evict_idle_now`, `evict_idle_for(config, now)`, `clear`) plus `lifecycle::global()` over `launcher::global()`.
- `operation` — `sandbox_operations(&card)`, `sandbox_operations_for_session(&card, Option<&str>)` (sync, static path only), `sandbox_operations_for_session_async(&card, session_id)`, `sandbox_operations_with_registries(&card, session_id, &providers, &lifecycle, now)`, `resolve_sandbox_endpoint(&card, session_id, &lifecycle, now)`, `register_builtin_providers(&registry, &gateway)`, `unsupported_sandbox(&str)`, and the forwarding wrappers `SandboxFsOperation`/`SandboxShellOperation`/`SandboxCodeOperation` (each exposes `endpoint()` and `isolation_key()`; `SandboxFsOperation` also `configured_conditional_write()`).
- `config` — `ContainerScope`, `OnStop`, `SandboxIsolationConfig`, `SandboxLauncherConfig`, `SandboxGatewayConfig`, `SandboxRunConfig`, `GatewayInvokeRequest`, `GatewayResponse`, `SandboxEndpoint`, `SESSION_ID_PLACEHOLDER`, `generate_isolation_key_template`, `resolve_isolation_key`, `isolation_key_template_from_config`.
- `http` (feature `sandbox`) — `HttpSandboxGateway { new, endpoint, config, invoke, probe }`, `HttpFsOperation`/`HttpShellOperation`/`HttpCodeOperation` (`new(gateway: Arc<HttpSandboxGateway>)`, `gateway()`), `install(&registry, sandbox_type, &endpoint, &config)`.

## Lifecycle State Machine

`ensure_endpoint` is the single entry point; it resolves the launcher for `launcher_config.launcher_type`, then runs `revive` (steps 1–2) and `launch_container` (steps 3–4):

1. A `running` record is reused and its `last_used_epoch_secs` refreshed (the launcher is not called).
2. A stored non-running record is checked: `check_status` `Running` → mark running; `Paused`/`Unknown` → `launcher.resume` then mark running; `Stopped` → drop the record and fall through; an error → 199008; a record without a `sandbox_id` falls through.
3. When `launcher_config.idle_ttl_seconds` is set, `evict_idle` runs before the launch.
4. `launcher.launch` runs, then the record is stored as `running`; if a concurrent caller already stored a running record for the key, the first record wins and the duplicate container is deleted (unless it reports the winner's `sandbox_id`).

`release(isolation_key, on_stop)`:

- `OnStop::Delete` → `launcher.delete`, record removed;
- `OnStop::Pause` → `launcher.pause`, record kept with status `paused` so the next `ensure_endpoint` resumes the same container;
- `OnStop::Keep` → no launcher call, record kept with a refreshed `last_used_epoch_secs`;
- an unknown key is a successful no-op, and a failing launcher is 199008 with the record restored so the call can be retried.

`evict_idle(ttl, now)` deletes every record whose `now - last_used >= ttl` regardless of `on_stop` (Python semantics) and returns the evicted keys; the comparison is `>=` so a zero TTL evicts everything (Python uses `>`).

## Constraints

- Time is injected: every lifecycle decision takes `now_epoch_secs` as an argument or reads the injected `Clock`; only `SystemClock`/`SandboxLifecycle::new`/`now_epoch_secs()` touch `SystemTime`, so tests never depend on the wall clock.
- `SandboxLifecycle` holds one `tokio::sync::Mutex<BTreeMap<String, SandboxRecord>>`; the lock is never held across a launcher await — the code clones what it needs, launches, then re-locks and inserts, keeping the first running record.
- Provider factories are synchronous `Fn`, not async: the endpoint comes from the lifecycle, so a factory only builds a client; anything async (handshake, token) belongs lazily inside the backend.
- Static path (unchanged behaviour): a non-blank `launcher_config.base_url` **or** `launcher_type == "pre_deploy"` builds the endpoint directly (`base_url`, `sandbox_id = None`, resolved isolation key) with no lifecycle bookkeeping, so existing `sandbox_operations*` callers keep working.
- Managed path: any other registered `launcher_type` needs the async lifecycle; the sync `sandbox_operations*` entry points return 199002 telling the caller to use `sandbox_operations_with_registries` or `SysOperation::sandbox_with_lifecycle`.
- `Error::Unsupported` is reserved for a genuinely missing transport: no provider registered for the `sandbox_type`, no launcher registered for the `launcher_type`, or the `sandbox` feature disabled; a launcher that exists but fails is always 199008 `SysOp`.
- `LauncherRegistry::register` rejects duplicates (`registry_message`, 199006); `register_builtin_launchers` is idempotent and never overwrites a host-registered `pre_deploy`.
- `LauncherRegistry`/`SandboxLifecycle` are cheap `Clone`s over shared `Arc` state; the process-global instances are `launcher::global()` and `lifecycle::global()`.
- `SandboxGatewayConfig::auth_headers`/`auth_query_params` are carried but never applied; `GatewayInvokeRequest`/`GatewayResponse` are the only wire-envelope types.
- `sandbox_types()` enumerates the filesystem map only, so a type registered with shell/code alone is not listed.
- `openjiuwen-core` (and therefore `openjiuwen-extensions`) does not forward the `sandbox` feature; the `http` module compiles only for a direct `openjiuwen-sysop` dependency with `features = ["sandbox"]`.
- Forwarding wrappers never serialize, inspect, or re-map parameters or `OpResult`; failures stay values with code 199007 and `"sandbox gateway error, operation: <method>, reason: <reason>"`.
- `register_builtin_providers` is inert unless the `sandbox` feature is on and `launcher.base_url` is non-blank; `install` checks `has_fs`/`has_shell`/`has_code` first, so a provider registered externally before construction wins, while a later duplicate `register_*` returns `ErrorKind::SysOp` ("already registered").
- Tests register fakes in local registries (`LauncherRegistry::new`, `SandboxProviderRegistry::new`) and never in `provider::global()`/`launcher::global()`; the only global assertions are read-only (`has("pre_deploy")`, identity, empty lifecycle sweep).

## Error-Code Mapping

| Failure | Code | Builder | Kind |
| --- | --- | --- | --- |
| card/gateway configuration (`launcher_config` missing, managed launcher via the sync path) | 199002 | `card::card_param_error` | `SysOp` |
| gateway transport failure | 199007 | `gateway_error` / `sandbox_gateway_message` | `SysOp` |
| launcher `launch`/`resume`/`delete`/`pause`/`check_status` failure | 199008 | `launcher_error` / `sandbox_launcher_message` | `SysOp` |
| provider failure | 199009 | `provider_error` / `sandbox_provider_message` | `SysOp` |
| blank isolation key | 199010 | `isolation_key_error` / `isolation_key_message` | `SysOp` |
| duplicate launcher registration | 199006 | `registry_message` | `SysOp` |
| unknown `launcher_type`, unknown `sandbox_type`, `sandbox` feature off | — | `unknown_launcher_error` / `unsupported_sandbox` | `Unsupported` |

## Known Issues / Deferred

- Launcher lifecycle is in-process only: `PreDeploymentLauncher` is the sole built-in, and the container transports (`aio`, `jiuwenbox`, `yuanrong`) are expected from `openjiuwen-extensions`, whose registration is not ported here (no network, no vendor SDKs, offline tests).
- `PreDeploymentLauncher::delete`/`pause`/`resume` are no-ops (Python tears shared-scope containers down through the extension providers); a managed launcher must override all four to be useful.
- Python's `sandbox_store` extras are not ported: `container_config_hash`, `metadata`, and the provider cache (providers are rebuilt from the resolved endpoint per operation).
- Endpoint discovery beyond `launch`'s `LaunchedSandbox`, auth-header/auth-query injection, and the streaming wire format remain unimplemented (`sandbox::http::invoke` is `Unsupported`, 199007).
- `evict_idle` compares with `>=` (Python uses `>`), and a failed eviction is logged rather than returned: the sweep keeps going so one broken container cannot pin the rest.
- No background sweeper: idle-stop runs on `ensure_endpoint` (when `idle_ttl_seconds` is set) and when a host calls `SysOperation::evict_idle_sandboxes*`.

## Routing Table

- `./mod.rs` → module declarations and the `sandbox` feature gate
- `./provider.rs` → factory type aliases, `SandboxProviderRegistry`, `global()`
- `./config.rs` → isolation/launcher/gateway/endpoint types and isolation-key helpers
- `./launcher.rs` → `SandboxLauncher` trait, `PreDeploymentLauncher`, `LauncherRegistry`, `global()`, error builders
- `./lifecycle/` → `SandboxRecord`, `Clock`/`FixedClock`, `SandboxLifecycle` (`ensure_endpoint`/`release`/`evict_idle`), `global()`; layout in `lifecycle/CONTEXT.md`
- `./operation.rs` → dispatch (`sandbox_operations*`, `resolve_sandbox_endpoint`, `register_builtin_providers`) and the three forwarding wrappers
- `./http.rs` → feature-gated `HttpSandboxGateway` (invoke = `Unsupported`, probe = the only network call), HTTP forwarding backends, `install`
