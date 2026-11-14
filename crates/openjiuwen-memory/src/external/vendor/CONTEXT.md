# openjiuwen-memory/src/external/vendor — one adapter per memory vendor

## Intent

The five vendor memory-provider adapters, split out of the former single `vendor.rs`.
Four speak real REST through [`crate::external::rest::RestProvider`] (`mem0`, `jiuwen_memory`, `lakebase`, `openviking`); `agentarts` is SDK-only and has no HTTP path, so every operation is a typed `Error::Unsupported` whether or not an endpoint is configured.
Each file owns the vendor's static `WireSpec` (id, display name, missing dependency, capability flags, default auth scheme, static headers, fallback scope) and a handle that wraps the shared transport.
The modules are private; the five handles are re-exported from this module so callers keep using `openjiuwen_memory::external::{Mem0MemoryProvider, …}`.
This directory has no child directories.

## API Surface

- `implement_vendor_provider!($adapter, $field; $($rail:item)*)` — the delegation macro: emits the complete `impl MemoryProvider for $adapter`, forwarding `name`, `capabilities`, `initialize`, `add`, `search`, `get`, `update`, `delete`, `list`, `clear`, and `health` to `self.$field`, then splices the caller's rail items into the same impl block.
- `vendor_factories()` — the five `ProviderFactory` values in kind order `AgentArts, Jiuwen, LakeBase, Mem0, OpenViking`; each `build`s its own kind and returns `Error::Validation` on a kind mismatch.
- `Mem0MemoryProvider`, `JiuwenMemoryProvider`, `LakeBaseMemoryProvider`, `OpenVikingMemoryProvider` — REST handles: infallible `new(&ProviderConfig)`, `config()`, `kind()`, `spec()`, a redacting `Debug`, and the `MemoryProvider` impl from the macro.
- `AgentArtsMemoryProvider` — SDK-only handle: infallible `new`, `config()`, `kind()`, redacting `Debug`, and the macro impl over a private `SdkOnlyProvider`.
- `option_text(config, key)` / `scope_from_options(config)` — `pub(super)` helpers shared by every spec (`options["user_id"]`/`["scope_id"]`/`["session_id"]`/`["agent_id"]`).

## Constraints

- An endpoint-less REST handle is inert: `capabilities()` is `ProviderCapabilities::none()` and every SPI method returns the transport's `Unsupported` error naming `WireSpec::missing()` and stating that no `endpoint` is configured.
- A configured REST handle reports exactly its spec's capability flags; the SDK-only adapter always reports none.
- `Debug` prints the adapter type, provider id, missing-dependency token, and configured endpoint only — never the configuration or the API key.
- `agentarts` never speaks raw HTTP: the Python adapter drives `agentarts.sdk.memory.MemoryClient`, and no documented REST contract exists to reimplement.
- No new dependencies; errors are `openjiuwen_common::error::{Error, Result}`; no `unsafe`, no `println!`; public items carry docs (`missing_docs` is denied).
- Wire shapes (`*_call`/`*_result`, `health_call`) are intentionally left at the `WireSpec` `Unsupported` defaults until the per-vendor wire work lands.

## Routing Table

- `./mod.rs` → module docs, the delegation macro, the shared option/scope helpers, `vendor_factories()`, and the unit tests
- `./mem0.rs` → `Mem0Spec` (`Token` auth), `Mem0MemoryProvider`, `SPEC`
- `./jiuwen.rs` → `JiuwenSpec` (bearer auth, message history), `JiuwenMemoryProvider`, `SPEC`
- `./lakebase.rs` → `LakeBaseSpec` (bearer auth, pagination), `LakeBaseMemoryProvider`, `SPEC`
- `./openviking.rs` → `OpenVikingSpec` (`X-API-Key`, account/user/agent headers), `OpenVikingMemoryProvider`, `SPEC`
- `./agentarts.rs` → `SdkOnlyProvider` and `AgentArtsMemoryProvider` (typed `Unsupported` only)

## Tests

`mod.rs` keeps the original `vendor.rs` suite adapted to the new layout: the unconfigured surface of each vendor (name, no capabilities, every SPI method `Unsupported` naming the token/`does not provide`/vendor), configured capability flags equal to the spec's, endpoint failover ordering, `Debug` redaction, `vendor_factories()` order/build/kind mismatch, and agentarts staying `Unsupported` with an endpoint configured.
All are offline: no test connects to a configured endpoint.
