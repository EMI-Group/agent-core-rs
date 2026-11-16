# llm/routing — model-selection compiler, failover routing model, TPM/RPM limiter

## Intent

Turn a declarative, already-resolved model selection (one model or a model group with candidate routes) into runtime `Model` configuration, and route calls over the enabled routes of a group.
Ports `openjiuwen/core/foundation/llm/routing/` (`schema.py`, `compiler.py`) and replaces the optional `intelli_router` vendor package with an in-crate routing model.
Everything here is pure data plus one `Model` implementation: no database, config file, or network access during compilation, and resolving a `ModelSelection` id into routes stays in the calling layer.
Per-route `tpm`/`rpm` budgets and the group/route `timeout_seconds` are enforced, not merely recorded.

## API Surface

- `schema` — `ModelSelection`, `SelectionKind`, `ResolvedModel`, `ResolvedRoute`, `ResolvedModelGroup`, `ResolvedRoutingConfig`, `RoutingStrategy`, `CompiledModelSelection`, `ModelSelectionWire`.
- `compiler` — `compile_model_selection(&ModelSelectionWire)`, `compile_route(&ResolvedRoute, &Value)`, `resolve_model_group_routes(&ResolvedModelGroup)`, `merge_request_defaults(&[&Value])`, `INTELLI_ROUTER_PROVIDER`, `ROUTING_PLAN_KEY`.
- `limit` — `RateLimiter` (`new`, `system`, `clock`, `reserve`, `reconcile`, `usage`), `Clock` trait (`now`, `sleep`) with `SystemClock`, `LimitKey`, `Limits`, `Reservation`, `WindowUsage`, `RATE_LIMIT_WINDOW` (60 s).
- `model` — `RoutingModel` (`from_group`, `with_models`, `with_provider`, `with_clock`, `rate_limiter`, `routes`, `route_ids`, `route_count`, `routing_config`, `model_group_id`) implementing `openjiuwen_foundation::llm::model::Model`, plus `is_retryable_failure`.
- `mod.rs` re-exports every public item above and declares `#[cfg(test)] mod tests`.

## Behaviour

- Request defaults merge `route.model.request_defaults` < group `request_config` < `route.request_overrides`; nested objects merge recursively and `null` never unsets an earlier value.
- Filtering drops `null`, `model`/`model_name`, client-configuration keys, and compiler-owned keys (`deployment`, `messages`, `stream`).
- `client_options` maps `client_id`/`verify_ssl`/`timeout`/`stream_first_chunk_timeout`/`stream_idle_timeout`/`max_retries`/`headers` onto client fields; every other key (including `endpoint_profile`) stays in `ModelClientConfig::extra`.
- Unrecognized request keys (including a `reasoning` intent) stay in `ModelRequestConfig::extra`; the OpenAI-compatible request builder resolves the reasoning intent one layer lower.
- A group with no strategy collapses to its first enabled route; a group with a strategy compiles to `provider = "intelli_router"` and records the plan under `extra["intelli_router"]`.
- `num_retries` defaults to `enabled_routes - 1`; `RoutingModel` attempts at most `num_retries + 1` candidates and never cycles.
- Failover retries `Transport`/`Timeout`/`Llm`/`External` failures only; `Validation`/`Config`/`NotFound`/`Unsupported` and every other kind return immediately.
- Streaming fails over only until the first chunk reaches the caller; a later error is propagated.
- `tag-filtered` keeps candidates tagged via `strategy_kwargs.tag` (string or array; `fallback_tag` accepted as the Python spelling), falling back to the group `fallback_tag`; an empty filtered set is `Error::Validation`.
- Priorities (`ResolvedRoute::priority`, additive Rust field) sort candidates stably by descending priority only when at least one route declares one; otherwise declared list order is preserved.
- TPM/RPM are enforced per `LimitKey` = the resolved route's `provider/model`, so routes of the same provider model share one provider-side budget and different providers never share one; `None` limits mean no throttle and record no state.
- Both budgets are rolling 60 s windows: `rpm` counts admitted requests, `tpm` sums reserved-then-reconciled tokens; a request that cannot fit waits on the injected `Clock` until enough of the window expires.
- Prompt tokens are reserved from an `ApproxTokenCounter` estimate before the request and reconciled with the provider's reported total afterwards (on a stream, when the stream ends); a failed or abandoned request keeps its estimate.
- A request whose estimate alone exceeds `tpm` is admitted once the window is empty, because waiting could never make it fit.
- A budget that cannot free up inside the route/group deadline is `Error::timeout` naming TPM/RPM, and it is terminal (another route would only repeat the wait); a non-positive or non-finite `rpm`/`tpm` is `Error::Validation`.
- `RoutingModel::with_clock` installs a `Clock`; the limiter's state is per `RoutingModel` instance (no process-global limiter), so a new clock starts from an empty window.
- Every attempt is wrapped in `tokio::time::timeout` of the route's own `timeout`, falling back to the group `timeout_seconds` (the route value wins as the more specific one); no value means unbounded.
- A timed-out attempt is an ordinary retryable failure and fails over to the next candidate; when every candidate timed out the returned error is `Error::timeout` (wrapping the `all routes failed ... last: timeout (...)` message), otherwise it stays `Error::llm`.
- On the streaming path the deadline bounds the time to the first chunk only, since a chunk reaching the caller commits the route.

## Deferred

- No cross-request circuit breaker or cooldown state: a failing route is skipped only by the retry policy of the current call.
- No tag-based cost/latency scoring — `tag-filtered` is a filter only.
- `RoutingStrategy::Custom` names are rejected by the compiler and treated as `ordered-failover` by `RoutingModel`.
- Health-check/observability routing options and `extra_body` flattening are not ported.

## Constraints

- No new dependencies: `serde`/`serde_json`, `openjiuwen_common::{error, security}`, `async-stream`, `futures`, `parking_lot`, `tokio` (`time`, `sync`), `tracing`, `wiremock` (dev).
- Every public item is documented; `missing_docs`, `unreachable_pub`, and `clippy --all-targets -- -D warnings` must stay clean.
- Logging target is `openjiuwen_foundation::llm::routing`; route ids, model names, and limit keys are logged, credentials never.
- The limiter lock is never held across an `.await`.

## Routing Table

- `./schema.rs` → serializable selection, route, group, routing-config, and compiled-result types
- `./compiler.rs` → selection compilation, request-default merging, route ordering
- `./limit.rs` → `RateLimiter`/`Clock`/`LimitKey`/`Limits` over the rolling TPM/RPM window, with `ManualClock` for tests
- `./model.rs` → `RoutingModel` (ordered failover, tag pinning, streaming failover, retry classification, budget reservation, attempt deadlines)
- `./tests.rs` → offline schema/compiler/routing tests, with `wiremock` endpoints for failover and `StubModel` + `ManualClock` for throttling/deadline
