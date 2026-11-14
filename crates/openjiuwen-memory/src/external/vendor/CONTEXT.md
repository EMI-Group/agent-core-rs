# openjiuwen-memory/src/external/vendor — one adapter per memory vendor

## Intent

The five vendor memory-provider adapters, split one per directory.
Four speak real HTTP through [`RestProvider`](../rest.rs): `mem0`, `jiuwen_memory`, `lakebase`, `openviking`.
`agentarts` is SDK-only (`agentarts.sdk.memory.MemoryClient`) and has no documented REST contract, so every operation is a typed `Error::Unsupported` whether or not an endpoint is configured.
Each adapter owns the vendor's `WireSpec` (static identity plus the per-verb request/response shapes) and a handle that wraps the shared transport and supplies the vendor's tool-rail overrides.
The Rust handles keep the Python provider ids (`agentarts`, `jiuwen_memory`, `lakebase`, `mem0`, `openviking`) and are re-exported from this module, so callers use `openjiuwen_memory::external::{Mem0MemoryProvider, …}`.

## API Surface

- `implement_vendor_provider!($adapter, $field; $($rail:item)*)` — the delegation macro: emits the complete `impl MemoryProvider for $adapter`, forwarding `name`, `capabilities`, `initialize`, `add`, `search`, `get`, `update`, `delete`, `list`, `clear`, `health` to `self.$field`, then splices the caller's rail overrides (`get_tool_schemas`, `handle_tool_call`, `prefetch`, `sync_turn`, `system_prompt_block`, `on_session_end`, `default_scope`, `is_available`, `is_initialized`) into the same impl block.
- `vendor_factories()` — the five `ProviderFactory` values in kind order `AgentArts, Jiuwen, LakeBase, Mem0, OpenViking`; each builds its own kind and returns `Error::Validation` on a kind mismatch.
- `Mem0MemoryProvider`, `JiuwenMemoryProvider`, `LakeBaseMemoryProvider`, `OpenVikingMemoryProvider` — REST handles: infallible `new(&ProviderConfig)`, `config()`, `kind()`, `spec()`, a redacting `Debug`, and the macro-generated `MemoryProvider` impl.
- `AgentArtsMemoryProvider` — SDK-only handle: infallible `new`, `config()`, `kind()`, redacting `Debug`, and the macro impl over a private `SdkOnlyProvider`.
- `option_text(config, key)` / `scope_from_options(config)` — `pub(super)` helpers shared by every spec (`options["user_id"]`/`["scope_id"]`/`["session_id"]`/`["agent_id"]`).

## Wire and Rail per Vendor

| Vendor | Wire | Rail tools |
| --- | --- | --- |
| `mem0` | `POST /v1/memories/` (one `user` message, `infer: false`), `POST /v1/memories/search/`, `GET /v1/memories/`, `GET`/`PUT`/`DELETE /v1/memories/{id}/`, `GET /v1/ping/`; auth `Authorization: Token` | `mem0_profile`, `mem0_search`, `mem0_conclude` |
| `jiuwen_memory` | `POST /v1/search` (five-segment `context.scope`, `top_k ≤ 50`, `disclosure: "l2"`), `POST /v1/add` (top-level unit array response, `segments` joined, `system_metadata.infer`), `GET /healthz`; auth bearer. `get`/`update`/`delete`/`list`/`clear` are typed `Unsupported` — the Mem2 HTTP API has no such verbs | `mem2_search`, `mem2_add` |
| `lakebase` | `POST …/memory/bases/{base_id}/ingest`, `POST …/recall`, `GET …/memories` (offset/limit), `GET`/`DELETE …/memories/{id}`, `GET …/stats`; auth bearer; integer ids stringified at the boundary; `update` is typed `Unsupported` (no documented endpoint) | `lkb_memory_search`, `lkb_memory_add`, `lkb_memory_list`, `lkb_memory_get`, `lkb_memory_delete`, `lkb_memory_stats` (routed through `health`) |
| `openviking` | `POST /api/v1/search/find`, `POST /api/v1/sessions/{id}/messages` (one `[Remember — <category>]` text part), `GET /api/v1/content/read?uri=`, `GET /api/v1/fs/ls?uri=` (directories skipped), `GET /health`; `X-API-Key` plus `X-OpenViking-{Account,User,Agent}`; `delete`/`update`/`clear` are typed `Unsupported`. Rail-only endpoints: `fs/tree`, `fs/stat`, `content/abstract`, `content/overview`, `POST /api/v1/resources`, `POST …/sessions/{id}/commit` | `viking_search`, `viking_read`, `viking_browse`, `viking_remember`, `viking_add_resource`; overrides `sync_turn` and `on_session_end` |
| `agentarts` | none — the vendor SDK is required | `external_memory_search` (schema and prompt block only; every call ends in the typed `Unsupported`) |

## Constraints

- An endpoint-less REST handle is inert: `capabilities()` is `ProviderCapabilities::none()` and every SPI method returns the transport's `Unsupported` naming `WireSpec::missing()` and stating that no `endpoint` is configured.
- A configured REST handle reports exactly its spec's capability flags; the SDK-only adapter always reports none.
- `Debug` prints the adapter type, provider id, missing-dependency token, and configured endpoint only — never the configuration or the API key.
- Rail dispatch never raises: an unknown tool answers `{"error": "unknown tool: <name>"}`, a missing required argument answers `{"error": "Missing required parameter: …"}`, and a backend failure answers `{"error": …}` with a `tracing::debug!`. `prefetch` degrades to `""`.
- Vendor write semantics are preserved rather than normalised (mem0 `infer: false`, jiuwen's scope payload, LakeBase's type mapping, OpenViking's session writes).
- No new dependencies; errors are `openjiuwen_common::error::{Error, Result}`; no `unsafe`, no `println!`; public items carry docs (`missing_docs` is denied).
- Files stay ≲1000 lines, which is why each vendor may keep `mod.rs` + `spec.rs` + `tests.rs`; do not move a shared helper into a vendor file's siblings — put it in `mod.rs` (`pub(super)`) when several vendors need it.

## Routing Table

- `./mod.rs` → module docs, the delegation macro, `option_text`/`scope_from_options`, `vendor_factories()`, and the cross-vendor unit tests
- `./mem0/` → `Mem0Spec` (`Token` auth) + `Mem0MemoryProvider` + mem0 rail and tests
- `./jiuwen/` → `JiuwenSpec` (bearer, five-segment scope, unit-array add response) + `JiuwenMemoryProvider` + mem2 rail and tests
- `./lakebase/` → `LakeBaseSpec` (bearer, base paths, integer-id boundary, memory-type mapping) + `LakeBaseMemoryProvider` + `lkb_*` rail and tests
- `./openviking/` → `OpenVikingSpec` (`X-API-Key`, account/user/agent headers, session writes) + `OpenVikingMemoryProvider` + `viking_*` rail and tests
- `./agentarts.rs` → `SdkOnlyProvider` and `AgentArtsMemoryProvider` (typed `Unsupported` plus the search tool schema)

## Test Strategy

- 82 offline tests in this directory: `mod.rs` 6 (unconfigured surface per vendor, configured capability flags, failover ordering, `Debug` redaction, factory order/build/kind mismatch, agentarts with an endpoint), `mem0` 14, `jiuwen` 14, `lakebase` 16, `openviking` 30, `agentarts` 8.
- Every vendor suite drives its handle against a `wiremock::MockServer` configured as the endpoint: request method/path/query/body/header assertions, response mapping with threshold filtering and sorted truncation, `404` → `None`/`false`, retry after `503`, failover to a second server, pre-cancelled call → `Cancelled`, and the rail (schema shape, `handle_tool_call` happy/unknown path, `prefetch`, prompt block, `sync_turn`/`on_session_end` where overridden).
- No test reaches a real vendor: no API key, network, or wall-clock ordering is required.
