# openjiuwen-memory/src/external — memory-provider SPI, registry, vendor stubs

## Intent

The pluggable external memory-provider seam for the Rust rewrite of openJiuwen, porting the *shape* of Python `openjiuwen.core.memory.external` (provider ids, vendor set, lazy "missing SDK surfaces at first use" behaviour) onto this crate's record types.
The in-process engine (`crate::engine`) stays the default backend; a provider lets a deployment delegate the same record surface to a remote service, a vendor SDK, or a custom backend without changing callers.
The Python SPI is tool-rail oriented (`get_tool_schemas`, `handle_tool_call`, `prefetch`, `sync_turn`, `system_prompt_block`); the Rust SPI is record oriented (`add`, `search`, `get`, `update`, `delete`, `list`, `clear`, `health`) because it adapts `MemoryScope`/`MemoryRecord`/`NewMemory`/`SearchOptions` directly.
No network client exists in this crate: vendor transports are deferred and every vendor path is a typed `Error::Unsupported`, never a silent no-op.
This directory has no child directories; the four modules are listed in the Routing Table.

## API Surface

`mod.rs` re-exports every public item; the modules are also public (`external::provider`, `external::local`, `external::registry`, `external::vendor`) for deeper paths in docs and tests.

Provider SPI (`provider.rs`):
- `MemoryProvider` — `#[async_trait] Send + Sync` trait, all methods `&self`: `name() -> &str`, `capabilities() -> ProviderCapabilities`, `async initialize() -> Result<()>`, `async add(&MemoryScope, Vec<NewMemory>, &AddOptions) -> Result<AddOutcome>`, `async search(&MemoryScope, &str, &SearchOptions) -> Result<Vec<RecallHit>>`, `async get(&MemoryScope, &str) -> Result<Option<MemoryRecord>>`, `async update(&MemoryScope, &str, MemoryUpdate, &CallOptions) -> Result<MemoryRecord>`, `async delete(&MemoryScope, &str) -> Result<bool>`, `async list(&MemoryFilter) -> Result<Vec<MemoryRecord>>`, `async clear(&MemoryFilter) -> Result<usize>`, `async health() -> Result<ProviderHealth>`.
- `ProviderCapabilities { semantic_search, keyword_search, graph, variables, message_history, pagination, max_batch: Option<usize> }` — serde defaults are all `false`/`None`; `none()` is the default, `local(has_embedding)` is the engine's report (keyword recall + CRUD always, semantic recall only with an embedding model, graph/variables/messages/paging `false`).
- `ProviderHealth { healthy, detail: Option<String>, checked_at: DateTime<Utc> }` — serde snake_case, `checked_at` defaults to "now"; `healthy()`, `degraded(detail)`, `with_detail(...)`.

Local provider (`local.rs`):
- `EngineProvider` — adapts `Arc<MemoryEngine>`; `new(engine)` (name `local`), `with_name(engine, name)` (blank name falls back to `local`), `engine()`, `is_initialized()`. Every call forwards to the engine unchanged, so scope validation, dedupe, ranking, and cancellation keep engine semantics. `initialize()` is idempotent (`AtomicBool`); `health()` performs one `list` on the sentinel scope `("__health__", "__health__")` and reports `Ok(healthy/degraded)`.
- `LocalProviderFactory` — `ProviderFactory` for `ProviderKind::Local` over a shared engine; `new(engine)`, `with_name(name)`, `engine()`; `build` falls back to the factory name when `ProviderConfig::name` is blank and rejects a different kind with `Error::Validation`.

Registry (`registry.rs`):
- `ProviderKind` — `AgentArts`, `Jiuwen`, `LakeBase`, `Mem0`, `OpenViking`, `Local`, `Custom(String)`; serde names are the Python ids (`agentarts`, `jiuwen_memory`, `lakebase`, `mem0`, `openviking`, `local`); `from_name` also accepts `agent_arts`, `lake_base`, `open_viking`, `jiuwen`, `engine`, `openjiuwen` (case-insensitive, trimmed); unknown/blank names become `Custom(trimmed)` instead of failing; `as_str`, `BUILTIN`, `Display`, `FromStr` (infallible), `Default = Local`.
- `ProviderConfig { kind, name, endpoint: Option<String>, api_key: Option<String>, timeout: Option<Duration>, options: serde_json::Map<String, Value> }` — serde snake_case with container defaults; `timeout` serializes as whole milliseconds; builders `new`, `with_name`, `with_endpoint`, `with_api_key`, `with_timeout`, `with_option`, accessors `provider_name()` (blank name → kind id), `option(key)`; `redacted()` returns a copy with `api_key = "<REDACTED>"`; `Debug` redacts `api_key` and sensitive `options` keys (`api_key`, `apikey`, `access_key`, `secret`, `secret_key`, `token`, `password`, `authorization`). `Serialize` deliberately keeps the credential so configs round-trip.
- `ProviderFactory` — `kind() -> ProviderKind`, `build(&ProviderConfig) -> Result<Arc<dyn MemoryProvider>>`; synchronous and side-effect free (no I/O; constructs handles only).
- `ProviderRegistry` — `Clone` over `Arc<RwLock<RegistryState>>`: `new()`, `with_vendor_stubs()`, `with_local(engine)`, `register_factory(factory) -> Option<...>`, `register(name, provider) -> Option<...>`, `get(name)`, `names()` (sorted), `default_provider()`, `set_default(name)`, `build(config)`, `select(config)`.

Vendor stubs (`vendor.rs`):
- `AgentArtsMemoryProvider`, `JiuwenMemoryProvider`, `LakeBaseMemoryProvider`, `Mem0MemoryProvider`, `OpenVikingMemoryProvider` — each has infallible `new(&ProviderConfig)`, `config()`, `kind()`, a redacting `Debug`, and a `MemoryProvider` impl in which *every* method returns the same typed `Error::Unsupported`.
- `vendor_factories() -> Vec<Arc<dyn ProviderFactory>>` — the five `ProviderFactory` values registered by `ProviderRegistry::with_vendor_stubs`.
- Private `VendorProvider` is the single helper that owns the error text, the `none()` capabilities, and the failing method bodies; the adapters are generated by one `vendor_adapter!` macro so no boilerplate is duplicated.

## SPI Contract

- Every method takes `&self`; one `Arc<dyn MemoryProvider>` is shared across tasks and the provider owns its own synchronization.
- Every method is cancel-safe: dropping the future leaves the backend consistent (a write is fully applied or not applied). `add`/`search`/`update` carry `CallOptions` and are raced against cancellation/deadline; `get`/`delete`/`list`/`clear` are bounded single-statement operations with no per-call options in the SPI (`EngineProvider` uses the default, never-cancelled options).
- `initialize()` is idempotent; calling other methods before it is permitted for an already-constructed backend (the local provider), while a remote provider may reject it.
- Results are never silently truncated: a missing capability is `Error::Unsupported`, never an empty list or zero count.
- Error kinds: `Validation` for caller input (scope, record, filter, options, oversized batch), `NotFound` for `update` of an unknown id, `Unsupported` for an absent capability, `Cancelled`/`Timeout` for call state, and the backend's own kind (`Transport`, `Memory`, …) for backend failure.
- Capability-dependent `Unsupported`: `initialize` when the SDK/transport is absent; `search` when the requested `SearchMode` is not declared; `add` when the backend is read-only or a batch exceeds `max_batch`; `health` when nothing can be probed.
- `health()` returns `Ok(ProviderHealth)` for a reachable-but-degraded backend (the local engine logs the probe failure and reports `healthy: false`); `Err(Unsupported)` is reserved for a provider with no backend to probe at all (the vendor stubs).

## Registry Semantics

- One registry per application, injected as an `Arc`; no process-global mutable state. The `parking_lot::RwLock` is never held across an `.await` (all registry methods are synchronous).
- Duplicate registration **replaces**: `register_factory` and `register` both follow `HashMap::insert` and return the previous entry; last registration wins (documented and tested).
- The first registered provider becomes the default; later registrations leave it alone. `set_default(name)` returns `NotFound` for an unregistered name.
- `build(config)` is pure: it resolves the factory for `config.kind`, builds, and changes no registry state. A kind with no factory returns `Error::Unsupported` naming the kind.
- `select(config)` = `build` + `register(config.provider_name())` + make it default; it returns the provider.
- `with_vendor_stubs()` registers the five vendor factories only (no instances, no default). `with_local(engine)` adds the local factory and a registered `local` instance (the default) over that engine; building further `ProviderKind::Local` configs yields providers over the same engine.
- `Debug` prints factory kinds, provider names, and the default name only — no configs, no credentials.

## Vendor Stub Policy

- Construction never fails: an adapter is a typed handle that keeps its `ProviderConfig`. The missing dependency surfaces at `initialize()` and at every data operation, mirroring Python's lazy `_get_client()`.
- Every trait method returns the same message shape: ``the <Vendor> memory provider requires <missing SDK/transport>, which this build does not provide``, naming the dependency verbatim (`agentarts-sdk`, `jiuwen_memory`, `httpx.AsyncClient` for LakeBase, `mem0ai`, OpenViking API client).
- Capabilities are `ProviderCapabilities::none()`; `health()` is `Unsupported`; no empty lists, no zero counts, no `unimplemented!`, no panic.
- No HTTP client, vendor SDK, wire format, or endpoint is invented or embedded; the vendor factories validate that `config.kind` matches their kind and return `Validation` otherwise.
- `Debug` on an adapter prints only the adapter type, provider id, and missing dependency — never the config or credential.

## Constraints

- Dependencies are limited to the crate's existing set (`async-trait`, `serde`/`serde_json`, `tokio`/`tokio-util`, `tracing`, `parking_lot`, `chrono`, `openjiuwen-common`); adding a network client here is out of scope.
- Errors are `openjiuwen_common::error::{Error, Result}`; no panics on user input, no `unwrap`/`expect` outside tests, no `unsafe`.
- Public data types are `serde`-serializable, snake_case, tolerant of unknown/missing fields; configs round-trip through JSON.
- Secrets never reach `Debug`, `tracing`, or a `ProviderHealth` detail.
- `tracing` only (target `openjiuwen_memory::external`); no `println!`/`eprintln!`.
- Files stay around ≤1000 lines; all public items carry docs (`missing_docs` is denied by clippy `-D warnings`).

## Routing Table

- `./provider.rs` → `MemoryProvider` trait, `ProviderCapabilities`, `ProviderHealth`, SPI contract docs
- `./local.rs` → `EngineProvider` over `Arc<MemoryEngine>`, `LocalProviderFactory`
- `./registry.rs` → `ProviderKind`, `ProviderConfig` (redaction, millisecond timeout), `ProviderFactory`, `ProviderRegistry`
- `./vendor.rs` → `AgentArts`/`Jiuwen`/`LakeBase`/`mem0`/`OpenViking` handles, shared `VendorProvider`, `vendor_factories()`
- `./mod.rs` → module docs, public re-exports, end-to-end registry smoke test

## Test Strategy

- Unit tests live beside each module and are offline and deterministic: 32 tests under `external` (`cargo test -p openjiuwen-memory --lib external`).
- `provider.rs` — capability defaults/`local()` flags and serde, health serde (RFC 3339 `checked_at`), object safety and `Send + Sync` of `Arc<dyn MemoryProvider>` via a null provider.
- `local.rs` — full round trip against `LongTermMemory::in_memory` (add → search → get → update → list → delete → clear → health), shared state with the facade, capability/behavior consistency (keyword-only engine refuses `SearchMode::Semantic`), name fallback, factory build/kind mismatch, `Debug` without the engine.
- `registry.rs` — kind ids/aliases/unknown-`Custom`/serde, config defaults/builders/serde/timeout-millis, `Debug` and `redacted()` redaction, first-registration default, duplicate replace, `set_default` `NotFound`, factory replace, `build` purity vs `select`, vendor-stub and local convenience constructors, `Send + Sync` + clone-shared state.
- `vendor.rs` — provider ids, kind/config accessors, every trait method returns `Unsupported` naming the SDK, redacting `Debug`, the five factories and their kind-mismatch `Validation`.
- `mod.rs` — end-to-end: a registry serves the local engine and reports the `mem0` stub through `select`.
- No test needs the network, credentials, wall-clock ordering, or an external service.
- Gates: `cargo fmt -p openjiuwen-memory -- --check`, `cargo check -p openjiuwen-memory --offline`, `cargo test -p openjiuwen-memory --offline --lib external`, `cargo clippy -p openjiuwen-memory --all-targets --offline -- -D warnings`, `cargo doc -p openjiuwen-memory --no-deps` warning-free.

## Deferred (not implemented here)

- Real vendor transports and SDKs (AgentArts, Jiuwen server/SDK, LakeBase REST, mem0, OpenViking) including payload shaping, retries, circuit breakers, and session mapping.
- The Python tool-rail surface of `MemoryProvider`: `get_tool_schemas`, `handle_tool_call`, `prefetch`, `sync_turn`, `system_prompt_block`, `on_session_end`, and their `harness/rails/memory/*` consumers.
- Provider-side graph memory, variables, and message history (the SPI is record-only; those stay facade/engine concerns).
- Scope-level provider selection/config, provider pooling/failover, and an HTTP/JSON provider implementation.
