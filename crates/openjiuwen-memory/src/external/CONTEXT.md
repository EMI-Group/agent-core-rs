# openjiuwen-memory/src/external — memory-provider SPI, tool rail, REST transports, registry

## Intent

The pluggable external memory-provider seam for the Rust rewrite of openJiuwen, porting Python `openjiuwen.core.memory.external` onto this crate's record types.
The in-process engine (`crate::engine`) stays the default backend; a provider delegates the same record surface to a remote service, a vendor SDK, or a custom backend without changing callers.
The SPI is record oriented (`add`, `search`, `get`, `update`, `delete`, `list`, `clear`, `health`) and carries Python's tool rail (`get_tool_schemas`, `handle_tool_call`, `prefetch`, `sync_turn`, `system_prompt_block`, `on_session_end`, `is_available`, `is_initialized`) as provided trait methods over that record surface.
Four vendor adapters (`mem0`, `jiuwen_memory`, `lakebase`, `openviking`) speak real HTTP through one shared transport ([`rest.rs`](rest.rs)); `agentarts` stays SDK-only and is a typed `Unsupported` surface.
Everything is offline-testable with `wiremock`: no test talks to a real vendor.

## API Surface

`mod.rs` re-exports every public item; the modules are public too (`external::provider`, `external::rail`, `external::local`, `external::registry`, `external::rest`, `external::rest_json`, `external::vendor`).

Provider SPI (`provider.rs`):
- `MemoryProvider` — `#[async_trait] Send + Sync`, all methods `&self`: `name`, `capabilities`, `initialize`, `add`, `search`, `get`, `update`, `delete`, `list`, `clear`, `health`.
- `ProviderCapabilities { semantic_search, keyword_search, graph, variables, message_history, pagination, max_batch }` — `none()` is the conservative default, `local(has_embedding)` is the engine's report.
- `ProviderHealth { healthy, detail, checked_at }` — `healthy()`, `degraded(detail)`, `with_detail(..)`.
- Provided rail methods: `is_available()`, `is_initialized()` (`false`, overridden by `EngineProvider` and by `RestProvider` to report its probe state), `default_scope()` (`MemoryScope::default()`), `get_tool_schemas()`, `handle_tool_call(tool, args)`, `prefetch(query)`, `sync_turn(user_msg, assistant_msg)`, `system_prompt_block()` (`""`), `on_session_end(messages)` (no-op).
- The default rail dispatches `add_memory` / `search_memory` / `list_memories` / `delete_memory` against `default_scope()`; an unknown name answers `{"error": "unknown tool: <name>"}` and a backend failure answers `{"error": …}` (never a raised error), matching the Python rail.

Rail helpers (`rail.rs`, pure, no I/O):
- Tool names/limits: `ADD_MEMORY_TOOL`, `SEARCH_MEMORY_TOOL`, `LIST_MEMORIES_TOOL`, `DELETE_MEMORY_TOOL`, `DEFAULT_PREFETCH_TOP_K = 5`, `DEFAULT_PREFETCH_MAX_TOKENS = 512`.
- Schema builders: `tool_schema`, `string_property`, `integer_property`, `number_property`, `boolean_property`, `enum_property`, `string_array_property`, `default_tool_schemas`.
- Payload shaping: `tool_failure` (the single producer of a failed rail call's `{"error": …}` payload plus one debug log), `tool_error`, `tool_result`, `tool_ok_result`, `hit_json`, `record_json`, `search_results`, `memory_listing`, `stored_result`, `deleted_result`, `render_bullets`.
- Argument parsing: `parse_search_args`, `parse_add_args`, `parse_kind`, `args_object` (tolerant of a non-object), `string_arg`, `integer_arg`, `number_arg`, `is_available`.

Local provider (`local.rs`): `EngineProvider` over `Arc<MemoryEngine>` (`new`, `with_name`, `engine`, `is_initialized`), `LocalProviderFactory`; overrides `is_available`, `is_initialized`, and a non-empty `system_prompt_block`.

Registry (`registry.rs`): `ProviderKind` (`AgentArts`, `Jiuwen`, `LakeBase`, `Mem0`, `OpenViking`, `Local`, `Custom`), `ProviderConfig` (`endpoint`, `api_key`, `timeout`, `options`), `ProviderFactory`, `ProviderRegistry` (`new`, `with_vendor_stubs`, `with_local`, `register_factory`, `register`, `get`, `names`, `default_provider`, `set_default`, `build`, `select`).

Transport (`rest.rs` + `rest/{call,auth,wire}.rs` + `rest_json.rs`):
- `RestCall` (base-relative method/path/query/headers/body), `absolute_url`.
- `AuthScheme { None, Bearer, Token, Header, Query }`, `resolve_auth(default, config)`.
- `Op { Add, Search, Get, Update, Delete, List, Clear, Health }` (`as_str`, `is_idempotent`), `SpecContext`.
- `WireSpec` — one static spec per vendor: `id`/`display`/`missing`/`capabilities`/`default_auth`/`headers`/`default_scope`/`supports`, plus an `*_call`/`*_result` pair per record verb and `health_call`; every shape method defaults to a typed `Unsupported`.
- `WireSpec::next_list_call` is the optional paging hook: a spec that pages answers the request for the next page from the answer it was given, and defaults to `None` (one page).
- `RestProvider` — `from_config(spec, config)`, `spec`, `config`, `is_configured`, `endpoints`, `client`, and the full `MemoryProvider` impl.
- `rest_json` — `body_value`, `results_array`, `results_array_checked`, `field_str`/`field_f64`/`field_i64`/`field_f32`/`field_value`, `parse_timestamp`, `vendor_record`, `listing_record`, `vendor_hit`, `rank_hits`, `truncate_detail`, `error_detail`, `scope_id`/`user_id`.

Vendor adapters (`vendor/`): `Mem0MemoryProvider`, `JiuwenMemoryProvider`, `LakeBaseMemoryProvider`, `OpenVikingMemoryProvider`, `AgentArtsMemoryProvider`, plus `vendor_factories()`.

## Transport Contract

- **Endpoints**: `ProviderConfig::endpoint` first, then `options["failover_endpoints"]` (array of strings); a spec never sees a base URL, so failover is transport-only.
- **Auth**: resolved per config from the spec's vendor default overridden by `options["auth"]` (`none|bearer|token|header|query`), with `options["auth_header"]` / `options["auth_query_param"]`; an absent or blank `api_key` attaches nothing.
- **Cancellation**: `CallOptions::check`/`run` around every call; `add`, `search`, `update` carry caller options, the other verbs use the never-cancelled default.
- **Retries**: per-op loop over `RetryPolicy` (statuses `429`/`5xx`); only idempotent verbs retry (`Add` is attempted once per endpoint); `options["max_retries"]` (default 2) and `options["retry_backoff_ms"]` override the policy.
- **Failover**: a transport failure or an exhausted retryable status moves to the next endpoint; the first `2xx` wins; all-failed names the last URL and endpoint count.
- **Status mapping**: `404` → `NotFound` (`get`/`delete` return `Ok(None)`/`Ok(false)` instead), `401`/`403` → `Security`, `400`/`409`/`412`/`422` → `Validation` with the vendor detail, `429`/`5xx`/other → `Transport`.
- **Writes**: `add` sends one request per memory (every vendor API accepts one content per call) and concatenates the outcomes.
- **Clear**: a spec without `clear_call` falls back to `list` + one `delete` per id; a vendor with neither ends in its typed `Unsupported`.
- **Health**: `health_call` when the spec has one (`2xx` healthy, anything else `Ok(degraded)`), otherwise a `list` probe on the sentinel scope `("__health__", "__health__")`; a configured provider never fails `health`.
- **Initialization**: `initialize` runs that same probe but honours its verdict — `401`/`403` is `Error::Security` naming the credential, any other non-`2xx` is `Error::Transport` carrying the vendor detail, and the handle stays uninitialized on both; a spec with no health call initializes unverified.
- **Pagination**: `list` follows `WireSpec::next_list_call` page by page, bounded by `MAX_LIST_PAGES = 200`, and an unfinished walk is a typed error rather than a silent partial listing, so `list` and `clear`'s list-then-delete fallback see the whole collection.
- **Listing shape**: `mem0` and `lakebase` decode listings with `results_array_checked`, which reports a non-empty `2xx` body that carries no recognised row array as `Error::Validation` naming the URL and the answered keys, so a renamed wrapper or a proxy error page cannot read as an empty collection; `results_array` stays tolerant for write and point-read answers.
- **Listing rows**: `listing_record` skips a row that carries no memory content (with a warning naming the operation and the row's keys), and `openviking` additionally skips directory entries, so a listing never reports an invented id or an empty body; a point read or write answer still uses `vendor_record`, which fills the id in locally.
- **Recall scoring**: an absent vendor score stays absent — `rank_hits` keeps an unscored row whatever `threshold` is, ranks scored rows first by score with the unscored rows after them in vendor order, and only then applies `top_k`, so a vendor whose recall answer omits `score` still answers with its matches.
- **Unconfigured handles**: `from_config` never fails; a handle without any endpoint reports `ProviderCapabilities::none()` and every method returns `Unsupported` naming `WireSpec::missing()` plus ``no `endpoint` is configured``. This contract is asserted by `crates/openjiuwen-memory/tests/external_provider.rs` (read-only for this node).

## Vendor Matrix

| Vendor | Transport | Auth | HTTP operations | Typed `Unsupported` |
| --- | --- | --- | --- | --- |
| `mem0` | `https://api.mem0.ai`-shaped REST | `Authorization: Token` | `POST /v1/memories/`, `POST /v1/memories/search/`, paged `GET /v1/memories/`, `GET|PUT|DELETE /v1/memories/{id}/`, `GET /v1/ping/` | a listing `filter` field other than `user_id`/`agent_id` |
| `jiuwen_memory` | Mem2 server | `Authorization: Bearer` | `POST /v1/search`, `POST /v1/add`, `GET /healthz` | `get`, `update`, `delete`, `list`, `clear` (the documented Mem2 API has no such verbs) |
| `lakebase` | DBay REST (`.../api/v1` base) | `Authorization: Bearer` | `POST …/memory/bases/{base_id}/ingest`, `POST …/recall`, `GET …/memories`, `GET|DELETE …/memories/{id}`, `GET …/stats` | `update` (no documented endpoint) |
| `openviking` | OpenViking REST | `X-API-Key` + `X-OpenViking-{Account,User,Agent}` | `POST /api/v1/search/find`, `POST /api/v1/sessions/{id}/messages`, `GET /api/v1/content/read`, `GET /api/v1/fs/ls`, `GET /health`, rail-only `/api/v1/fs/{tree,stat}`, `/api/v1/content/{abstract,overview}`, `POST /api/v1/resources`, `POST /api/v1/sessions/{id}/commit` | `delete`, `update`, `clear` (no documented endpoints) |
| `agentarts` | none — `agentarts.sdk.memory.MemoryClient` | — | — | every operation, configured or not |

Per-vendor rail: `mem0_*` (3 tools), `mem2_*` (2), `lkb_*` (6), `viking_*` (5), `external_memory_search` (1), each with its Python prompt block, `prefetch` heading and payload shapes.
`openviking` also overrides `sync_turn` (two session messages) and `on_session_end` (session commit); `lakebase` overrides `sync_turn` (episode ingest).

Per-vendor decode rules:
- `mem0` pages `GET /v1/memories/` through `next_list_call` (a short page ends the walk) and rejects a listing `filter` it cannot express with a typed `Unsupported` naming the fields, because the endpoint accepts only `user_id` and `agent_id`.
- `jiuwen_memory` propagates a non-JSON recall body as a typed error instead of reading it as "no hits", and reports an accepted write whose answer is not a top-level JSON array as **added**; only an empty unit array is a dedupe skip.
- `lakebase` warns and reports a client-generated id when an ingest answer carries no `memory_id`, and sends a multi-kind `filter.kinds` listing unfiltered because the API takes one `memory_type`.
- `openviking` decodes its directory listing from `result`, skipping directories and entries with neither a `uri` nor content.

## Constraints

- Dependencies stay at the crate's existing set (`async-trait`, `serde`/`serde_json`, `tokio`/`tokio-util`, `tracing`, `parking_lot`, `chrono`, `openjiuwen-common`); HTTP goes through `openjiuwen_common::http` (no direct `reqwest` dependency).
- Errors are `openjiuwen_common::error::{Error, Result}`; no panics on user input, no `unwrap`/`expect` outside tests, no `unsafe`.
- Public data types are `serde`-serializable, snake_case, tolerant of unknown/missing fields.
- Secrets never reach `Debug`, `tracing`, or an error message: `ProviderConfig` redacts the key, and every vendor `Debug` prints identity plus endpoint only.
- `tracing` only (targets `openjiuwen_memory::external…`); no `println!`/`eprintln!`.
- Files stay around ≤1000 lines; vendor adapters may split into `vendor/<vendor>/{mod,spec,tests}.rs`.
- All public items documented (`missing_docs` denied by clippy `-D warnings`).
- Do not edit `crates/openjiuwen-memory/tests/external_provider.rs` from this node; it pins the unconfigured-handle contract.

## Design Decisions

- **One `WireSpec` per vendor, one `RestProvider` for all.** Validation, auth, retries, failover, cancellation, status mapping and error typing live once; a vendor file is declarative wire shape plus its rail overrides (the `implement_vendor_provider!` macro forwards the eleven required methods and splices the overrides).
- **`endpoint` is the switch.** A configured handle speaks HTTP; an endpoint-less handle is inert with a typed reason, so a registry can build providers from a draft configuration exactly like the Python adapters' lazy client acquisition.
- **`agentarts` has no HTTP fallback.** The Python adapter drives a vendor SDK and never constructs a REST path, so inventing a wire format here would be a fabrication; the handle is a typed handle and its rail is schema/prompt-complete.
- **Vendor write semantics are preserved, not normalised.** `infer: false` for mem0 (records are stored verbatim), jiuwen's five-segment scope payload and `system_metadata.infer`, LakeBase's integer ids and six memory types, OpenViking's session-message writes and commit-on-session-end.
- **Tolerant decoding.** Unknown fields are ignored, `results`/`memories`/`items`/`data`/bare arrays are all accepted, timestamps accept RFC 3339 and unix seconds, and a missing id is generated locally rather than failing the read.
- **Rail is inherited, not required.** A custom backend implements only the record SPI and still gets a working four-tool rail, `prefetch`, and `sync_turn`.
- **Retry/failover are per request.** There is no provider-level circuit breaker, health scoring, or load balancing; `options["failover_endpoints"]` is an ordered list, first success wins.

## Known Issues

- `clear` on `jiuwen_memory` and `openviking` runs the transport's list-then-delete fallback and therefore ends in the vendor's typed `Unsupported` (nothing is removed); the same is true for a `lakebase`/`openviking` clear where `delete` is supported but `list` is not.
- `jiuwen_memory`'s write decoder overwrites the decoded content with its `segments` join, so a unit without `segments` yields an `added` record with empty content (and `mem2_add` answers `"stored"` with `content: ""`).
- `vendor_record` fills in a local id when a vendor row carries none, so an id-less row from a point read or a listing decodes to a record whose id cannot address it remotely.
- `lakebase` reports `added` from any `2xx` ingest even without a `memory_id` (warning only), and sends a multi-kind `filter.kinds` listing unfiltered.
- `SearchMode::Hybrid` is not validated against `ProviderCapabilities`, so a semantic-only vendor (`mem0`, `openviking`) answers a hybrid request with its semantic result instead of a typed `Unsupported`.
- `OpenVikingSpec::add_result` and the `viking_add_resource` rail tool report success from a `2xx` without decoding it (a client-generated id, an empty `root_uri`), which is deliberate for session-message writes but loses any vendor rejection carried in a `2xx` body.
- `openviking`'s directory listing decodes a body that carries no `result` array as an empty listing rather than a shape-drift error.
- `lakebase`'s `list_call` asks for one `offset`/`limit` page and the spec implements no `next_list_call`, so `list` returns at most `options["limit"]` (20 by default) records and `clear`'s list-then-delete fallback deletes only that page and reports success while the rest of the collection remains, despite the spec declaring `pagination: true`.
- `openviking`'s rail requests (`viking_*`, `prefetch`, `sync_turn`, `on_session_end`) run through `send_rail`, which uses only the first endpoint and the client directly, so they get no endpoint failover, no retry on `429`/`5xx`, and no cancellation while every record verb does.
- `status_error` and `probe_error` insert `error_detail`'s vendor-supplied message into the error text untruncated; the documented 512-character cap applies only to the raw-body fallback (`truncate_detail`).
- The default `add_memory` tool schema offers all seven `MemoryType` names, including `variable` and `unknown`, which `NewMemory::validate` rejects, so a model following the schema can produce a call that always fails.
- `openviking` interpolates the session id into `/api/v1/sessions/{id}/messages` and `/api/v1/sessions/{id}/commit` without percent-encoding, so an id containing `/`, `?` or `#` addresses a different request than the one intended.
- `ProviderRegistry::select` does not make the provider it built the default when one is already registered, contradicting the doc comment on `select` and the registry module docs.

## Deferred (not implemented here)

- `agentarts`: the real SDK transport (`agentarts.sdk.memory.MemoryClient`), session mapping KV store, and its `MemorySearchFilter` strategies.
- `mem0`: the SDK's `rerank` flag, `queue_prefetch` warm-up task, and the Python 5-failure/120 s circuit breaker.
- `jiuwen_memory`: the in-process `jiuwen_memory` SDK backend (`mode="sdk"`) and the circuit breaker; server verbs beyond `/v1/search`, `/v1/add`, `/healthz`.
- `lakebase`: branch/version endpoints and their 11 rail tools, `digest`, `traits`, `stats`, `switch_base` (the rail's `lkb_memory_stats` routes through `health` instead).
- `openviking`: nothing endpoint-wise is missing from the ported rail; Python's `title`-ignored `viking_add_resource` drift is reproduced deliberately.
- The `harness/rails/memory/*` consumers: the rail exists and is exercised by unit tests, but no agent/harness code drives `prefetch`/`sync_turn`/`on_session_end` yet.
- Provider-side graph memory, variables, and message history (the SPI is record-only; those stay facade/engine concerns).
- Scope-level provider selection, provider pooling/load balancing, and live-API contract tests (everything is `wiremock`).

## Routing Table

- `./provider.rs` → `MemoryProvider` (record SPI + provided rail methods), `ProviderCapabilities`, `ProviderHealth`, SPI/rail contract docs
- `./rail.rs` → pure rail helpers: schema builders, default four-tool set, argument parsing, payload shaping, Markdown rendering
- `./local.rs` → `EngineProvider` over `Arc<MemoryEngine>`, `LocalProviderFactory`
- `./registry.rs` → `ProviderKind`, `ProviderConfig` (redaction, millisecond timeout), `ProviderFactory`, `ProviderRegistry`
- `./rest.rs` → `RestProvider`, `WireSpec`, `Op`, `SpecContext`, the transport contract (`rest/CONTEXT.md`)
- `./rest/` → `call.rs` (`RestCall`), `auth.rs` (`AuthScheme`, `resolve_auth`), `wire.rs` (`WireSpec`) (`rest/CONTEXT.md`)
- `./rest_json.rs` → tolerant vendor-body accessors and record/hit mapping
- `./vendor/` → the five adapter handles, `vendor_factories()`, the delegation macro (`vendor/CONTEXT.md`)
- `./mod.rs` → module docs, public re-exports, registry smoke test

## Test Strategy

- 168 offline unit tests under `external::` (`cargo test -p openjiuwen-memory --offline --lib external`).
- `rest/tests.rs` (19) — wiremock: auth header per scheme, add/search/get/delete/list bodies, `404` → `None`/`false`, `401` → `Security`, `422` → `Validation`, `503` → retry then success, failover to a second server, all-failed names the last endpoint, pre-cancelled options → `Cancelled`, and the unconfigured-handle message.
- `rest_json.rs` (13) — array/alias/timestamp decoding, `vendor_record`/`vendor_hit` fallbacks, `rank_hits` threshold/ordering with scored and unscored rows, `results_array_checked` shape drift, and `listing_record` skipping.
- `provider.rs` (11) + `rail.rs` (9) + `local.rs` (9) — rail defaults through an unbacked provider, schema shape, argument bounds, `EngineProvider` rail round trip.
- `registry.rs` (14) + `mod.rs` (1) + `vendor/mod.rs` (6) — kinds/configs/redaction/registry semantics, the unconfigured surface of every vendor, configured capability flags, failover ordering, factory order and kind mismatch.
- Per vendor: `mem0` (14), `jiuwen` (16), `lakebase` (16), `openviking` (32), `agentarts` (8) — request method/path/body/auth assertions, response mapping with threshold filtering and unscored rows, `404` handling, retry, failover, cancellation, listing pagination and shape drift, capability flags, and the rail (schemas, `handle_tool_call` happy/unknown path, `prefetch`, prompt block, `sync_turn`/`on_session_end`).
- Everything runs offline through `wiremock`; no API key, network, or wall-clock ordering is needed.
- Gates: `cargo fmt -p openjiuwen-memory -- --check`, `cargo test -p openjiuwen-memory --offline --all-features`, `cargo clippy -p openjiuwen-memory --all-targets --all-features --offline -- -D warnings`, `cargo doc -p openjiuwen-memory --no-deps`.
