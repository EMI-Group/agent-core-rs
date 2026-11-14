# openjiuwen-memory/src/external/rest — the shared vendor REST transport

## Intent

The single HTTP transport every vendor memory adapter runs on: a vendor contributes a declarative [`WireSpec`](wire.rs) and [`RestProvider`](mod.rs) owns validation, authentication, cancellation, retries, failover, status mapping and error typing.
A spec never sees a base URL — it returns a base-relative [`RestCall`](call.rs) — which is what makes endpoint failover possible without vendor knowledge.
The transport is internal infrastructure: it is public for documentation and tests, but callers use the vendor handles in [`super::vendor`](../vendor/CONTEXT.md).

## API Surface

`rest.rs` (module root) re-exports `RestCall`, `Op`, `SpecContext`, `WireSpec`, `AuthScheme`, `resolve_auth`, and the tolerant JSON accessors from [`super::rest_json`](../rest_json.rs).
- `RestProvider` — `from_config(spec: &'static dyn WireSpec, config: &ProviderConfig)`, `spec()`, `config()`, `is_configured()`, `endpoints()`, `client() -> Option<&HttpClient>`; implements `MemoryProvider` completely (record verbs + inherited rail).
- Constants: `DEFAULT_TIMEOUT = 30 s`, `DEFAULT_MAX_RETRIES = 2`, `DEFAULT_RETRY_BACKOFF = 200 ms`, `MAX_RETRY_BACKOFF = 5 s`, `BODY_SNIPPET_CHARS = 512`, `USER_AGENT = "openjiuwen-memory/<version>"`.
- Log target: `openjiuwen_memory::external::rest` (retries, failover, degraded health); never credentials.

`rest/call.rs` — `RestCall { method: Method, path, query, headers, body }` with `get`/`post`/`put`/`patch`/`delete` constructors, `query`/`queries`/`header`/`json`, and `absolute_url(base, path)` (joins without doubling or dropping `/`).

`rest/auth.rs` — `AuthScheme { None, Bearer, Token, Header { name }, Query { name } }` and `resolve_auth(default, config)` reading `options["auth"]` (`none`/`bearer`/`token`/`header`/`query`), `options["auth_header"]`, `options["auth_query_param"]`; unknown/blank values fall back to the vendor default, and an absent/blank `api_key` attaches nothing.

`rest/wire.rs` — `Op { Add, Search, Get, Update, Delete, List, Clear, Health }` (`as_str`, `is_idempotent`), `SpecContext { config, scope }` (with `user_id`/`scope_id`/`session_id`/`agent_id`/`option` accessors), and the `WireSpec` trait:
- Static: `id`, `display`, `missing`, `capabilities`, `default_auth`, `auth(config)`, `headers(config)`, `default_scope(config)`, `supports(op)`, `unsupported(op)`.
- Shape pairs, each defaulting to `Err(self.unsupported(op))`: `add_call`/`add_result`, `search_call`/`search_result`, `get_call`/`get_result`, `update_call`/`update_result`, `delete_call`/`delete_result`, `list_call`/`list_result`, `clear_call`/`clear_result`, `health_call`.

## Transport Contract

- **Order of work per call**: validate (`MemoryScope::validate`, empty `MemoryUpdate` → `Validation`), check cancellation, build the spec's `RestCall`, then run the endpoint/retry loop.
- **Endpoints**: `config.endpoint` first, then `options["failover_endpoints"]` (blank entries and trailing slashes normalised).
- **Auth** is resolved once at construction and injected into every request (header or query, per `AuthScheme`).
- **Retries**: only for `op.is_idempotent()` verbs, on `Error::is_retryable()` transport/timeout errors and on retryable statuses (`429`, `5xx`); `options["max_retries"]` and `options["retry_backoff_ms"]` override the policy; the sleep and the request both race the cancellation token.
- **Failover**: a final transport-level failure (or an exhausted retryable status) moves to the next endpoint; a definitive HTTP answer never fails over; the first `2xx` wins and an all-failed run returns the last error naming the endpoint and attempt count.
- **Status mapping** (`status_error`): `404` → `NotFound`, `401`/`403` → `Security`, `400`/`409`/`412`/`422` → `Validation` with the vendor detail (from `error`/`message`/`detail`/`errors[0].message`, truncated to 512 chars), `429`/`5xx`/other non-2xx → `Transport`. `get`/`delete` translate `404` into `Ok(None)`/`Ok(false)` before mapping.
- **Writes**: `add` makes one request per `NewMemory` (every vendor API accepts one content per call) and concatenates the outcomes; the spec always sees a one-element slice.
- **Clear**: a spec that does not support `Op::Clear` is served by `list` + one `delete` per id; delete failures are not counted.
- **Health**: `health_call` when implemented (`2xx` healthy, any other status or a transport failure → `Ok(degraded)`), otherwise a `list` probe on the sentinel scope `("__health__", "__health__")`; a configured provider never returns `Err` from `health`.
- **`initialize`**: idempotent (`AtomicBool`); probes the health call when the spec has one, and treats any HTTP answer as success — only a transport failure is an error.
- **Unconfigured handles**: with no endpoint at all, `capabilities()` is `none()` and every method returns `Unsupported` formatted as ``the <display> memory provider requires <missing>, which this build does not provide: no `endpoint` is configured``. `crates/openjiuwen-memory/tests/external_provider.rs` pins this text.

## Constraints

- HTTP goes through `openjiuwen_common::http` only (`HttpClient`, `HttpClientConfig`, `HttpRequest`, `HttpResponse`, `RetryPolicy`, `Method`); this crate has no direct `reqwest` dependency.
- `from_config` never fails: an environmental client-construction failure (e.g. a malformed proxy) is stored and surfaced as a typed transport error from every call, never swallowed.
- Request bodies are JSON via `RestCall::json`; the transport sets `content-type: application/json` for them.
- `Debug` on `RestProvider` prints identity and endpoint, never the credential or config.
- Errors are `openjiuwen_common::error::{Error, Result}`; no panics on user input; `tracing` only.
- Vendors must not edit this module to add a shared helper by hand — a genuinely shared helper belongs here, a vendor-specific one stays in the vendor file.

## Routing Table

- `./call.rs` → `RestCall` and `absolute_url`
- `./auth.rs` → `AuthScheme` and `resolve_auth`
- `./wire.rs` → `Op`, `SpecContext`, `WireSpec`
- `./tests.rs` → the transport suite (wiremock)
- `../rest_json.rs` → tolerant JSON accessors, `vendor_record`, `vendor_hit`

## Test Strategy

- 19 tests in `rest/tests.rs`, all offline with `wiremock`: a `TestSpec` covers the unconfigured message and `none()` capabilities, configured capability flags, add/search/get/delete paths and bodies, the auth header reaching the wire, `401`/`422`/`404` mapping, `503` → retry → success (with `received_requests` counts), failover to a second server, all-failed error text, a pre-cancelled call reaching no server, and the `resolve_auth` variants.
- `rest_json` has 9 companion tests (arrays/aliases/timestamps/id generation).
- Gates: `cargo fmt -p openjiuwen-memory -- --check`, `cargo test -p openjiuwen-memory --offline --lib external::rest`, `cargo clippy -p openjiuwen-memory --all-targets --all-features --offline -- -D warnings`, `cargo doc -p openjiuwen-memory --no-deps`.
