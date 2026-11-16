# openjiuwen-agent-teams/src/models

## Intent

Model pool and per-request router for a team: the Rust home of the pool data structure and the request-time routing that Python splits across `openjiuwen/agent_teams/models/pool.py` and `models/allocator.py` (`RouterAllocator`, `ByModelNameAllocator`, `RoundRobinModelAllocator`).
`ModelPool` is the ordered, validated list of LLM endpoints a team declares; pool order is an entry's identity, so it is never re-sorted.
`ModelRouter` chooses one pooled endpoint per request — by requested name, rotation, least in-flight load, or an ordered rule list — and reports a fallback chain for retry.
Build-time per-member assignment is **not** here: it lives in `../runtime/allocator.rs` (`ModelAllocator`), and the two compose rather than replace each other.
Everything is offline and in-process: no vendor SDK, no network, no persisted counters.

## API Surface

```rust
// pool.rs
pub struct ModelPool;                                  // Debug, Clone, PartialEq, Default (empty)
impl ModelPool {
    pub fn new(entries: Vec<ModelPoolEntry>) -> Result<Self>;      // validates blank/duplicate names
    pub fn from_entries(entries: &[ModelPoolEntry]) -> Result<Self>;
    pub fn empty() -> Self;
    pub fn is_empty(&self) -> bool;      pub fn len(&self) -> usize;
    pub fn entries(&self) -> &[ModelPoolEntry];
    pub fn get(&self, index: usize) -> Option<&ModelPoolEntry>;
    pub fn find(&self, model_name: &str) -> Option<(usize, &ModelPoolEntry)>;
    pub fn contains(&self, model_name: &str) -> bool;
    pub fn model_names(&self) -> Vec<&str>;                       // pool order
    pub fn model_group_label(&self) -> String;                    // Python model_group_label semantics
}
// router.rs — strategy + hint
#[derive(Debug, Clone, Copy, Default /* RoundRobin */, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RouterStrategy { ByName, RoundRobin, LeastBusy, RuleBased }   // by_name | round_robin | least_busy | rule_based
impl RouterStrategy { pub const fn as_str(self) -> &'static str; }
impl From<ModelPoolStrategy> for RouterStrategy;   // RoundRobin -> RoundRobin, ByModelName -> ByName, IntelliRouter -> RoundRobin
pub enum Complexity { Low, Medium, High }                             // Ord, snake_case
pub struct RequestHint { pub model_name: Option<String>, pub role: Option<TeamRole>,
                         pub task_kind: Option<String>, pub complexity: Option<Complexity>,
                         pub needs_tools: bool }                      // Debug, Clone, Default, PartialEq, Eq, serde
impl RequestHint {
    pub fn new() -> Self;   pub fn for_role(role: TeamRole) -> Self;
    pub fn with_model_name(impl Into<String>) -> Self;   pub fn with_role(TeamRole) -> Self;
    pub fn with_task_kind(impl Into<String>) -> Self;    pub fn with_complexity(Complexity) -> Self;
    pub fn with_needs_tools(bool) -> Self;
}
// router.rs — rules
pub struct RuleMatcher { pub role: Option<TeamRole>, pub task_kind: Option<String>,
                         pub complexity: Option<Complexity>, pub min_complexity: Option<Complexity>,
                         pub needs_tools: Option<bool> }               // every Some field must match
impl RuleMatcher {
    pub fn matches(&self, hint: &RequestHint) -> bool;
    pub fn with_role/with_task_kind/with_complexity/with_min_complexity/with_needs_tools(self, ..) -> Self;
}
pub struct RoutingRule { pub name: String, pub priority: u32, pub matcher: RuleMatcher, pub target: String }
impl RoutingRule { pub fn new(name, matcher, target) -> Self;      // priority 0
                   pub fn with_priority(self, u32) -> Self;   pub fn matches(&self, &RequestHint) -> bool; }
// router.rs — decision + router
pub enum RouteReason { RequestedName, Rule, RoundRobin, LeastBusy, FirstEntry }   // snake_case, const as_str()
pub struct RouterDecision { pub entry: ModelPoolEntry, pub index: usize, pub reason: RouteReason,
                            pub rule: Option<String>, pub fallback_chain: Vec<String> }  // Debug, Clone, Default, PartialEq, serde
pub struct ModelRouter;   // pool + strategy + sorted rules + Mutex<state{cursor, in_flight per index}>
impl ModelRouter {
    pub fn new(pool: ModelPool, strategy: RouterStrategy) -> Self;
    pub fn for_pool(entries: &[ModelPoolEntry], strategy: ModelPoolStrategy) -> Result<Self>;
    pub fn with_rules(self, rules: Vec<RoutingRule>) -> Self;      // sorts (priority asc, name asc)
    pub fn pool(&self) -> &ModelPool;   pub const fn strategy(&self) -> RouterStrategy;
    pub fn rules(&self) -> &[RoutingRule];
    pub fn decide(&self, hint: &RequestHint) -> Result<RouterDecision>;       // preview: no state change
    pub fn acquire(&self, hint: &RequestHint) -> Result<RouteGuard<'_>>;      // atomic choose + reserve
    pub fn in_flight(&self) -> Vec<usize>;   pub fn total_in_flight(&self) -> usize;
}
pub struct RouteGuard<'a>;                       // Debug
impl RouteGuard<'_> { pub fn decision(&self) -> &RouterDecision; pub fn entry(&self) -> &ModelPoolEntry; }
impl Drop for RouteGuard<'_>;                    // decrements in-flight, saturating, never panics
```

`mod.rs` re-exports exactly: `ModelPool`, `Complexity`, `ModelRouter`, `RequestHint`, `RouteGuard`, `RouteReason`, `RouterDecision`, `RouterStrategy`, `RoutingRule`, `RuleMatcher`.
The crate root does not re-export this module's types yet (a flat re-export at `src/lib.rs` belongs to the crate-root node).

## Constraints

- Only `src/models/**` is edited from this node; `src/lib.rs`, `Cargo.toml`, other `src/` directories, `tests/` and other crates are read-only, and nothing imports `../kv_cache/`, `../observability/` or `../monitor/`.
- Dependencies are `openjiuwen_common::error::{Error, Result}`, `parking_lot::Mutex`, `serde`, `serde_json` and `crate::schema`; no new crates, no network, no filesystem.
- Errors are typed: `Error::validation` for a malformed pool, `Error::config` when no endpoint is configured or a pool index is unreachable; no user input panics.
- Public data types are `Debug, Clone` and serde snake_case with container-level `default`, so unknown fields from a newer host are tolerated.
- `Debug` on `ModelPool`, `ModelRouter` and `RouteGuard` never shows credentials or endpoint URLs — only counts and model names (`ModelPoolEntry::api_key` is a `SecretString`, but the pool keeps it out of the output anyway).
- `no unsafe`, no global state: every router owns its own `Mutex<state>`, so tests construct local instances.
- The lock is never held across an `.await`; `decide`/`acquire` are synchronous and finish in microseconds.
- Files ≤1000 lines with `#[cfg(test)] mod tests` beside the code.

## Design Decisions

### Routing semantics

- Empty pool: `decide` and `acquire` both return `Error::config("no model configured for requests; set model_pool on the team spec")`; nothing is reserved.
- `ByName`: the entry whose `model_name` equals `RequestHint::model_name` wins with `RouteReason::RequestedName`; an absent name or a name the pool does not serve falls back to pool index 0 with `RouteReason::FirstEntry`.
- `RoundRobin`: rotates through pool order and ignores the requested name; `decide` reports the current cursor without advancing it, `acquire` advances it, so N sequential acquisitions over k entries yield the repeating pattern `0..k`.
- `LeastBusy`: picks the smallest in-flight count, tie-broken by the lowest pool index; counts come only from live `RouteGuard`s, so a preview sees the same loads an acquire would.
- `RuleBased`: rules are sorted by `(priority asc, name asc)` (a lower priority number wins, the name makes the order total regardless of declaration order); the first matching rule whose target is pooled wins with `RouteReason::Rule` and `rule = Some(name)`.
- A rule whose target is missing from the pool is skipped, not fatal: the pool may have been refreshed under an unchanged rule set, and the next matching rule still yields a usable endpoint.
- With no matching rule, `RuleBased` falls back to `ByName` semantics (a pooled requested name, else pool index 0) — documented choice, so rule-based teams keep honouring explicit model requests.
- Rules attached to a non-rule strategy are reported through `ModelRouter::rules` but never consulted.

### Fallback chain

- `fallback_chain` lists **model names** (not indices) the caller should try next if the chosen endpoint fails, in this order: the remaining matching rule targets after the fired rule (rule-based only), then the strategy's own ordering of the other entries, then the first entry.
- The strategy ordering is rotation order after the chosen position for `RoundRobin`, `(in-flight asc, index asc)` for `LeastBusy`, and pool order for `ByName`/`RuleBased`.
- The chain is de-duplicated, contains only names the pool actually serves, and never contains the chosen model name.
- A single-entry pool therefore yields an empty chain, and a by-name decision that fell back to index 0 lists the other entries but never the requested-but-unpooled name (it is not an endpoint).

### In-flight accounting

- `acquire` reserves exactly one slot on the chosen index under the same lock that made the choice, so two concurrent acquires cannot both see the same zero load.
- Reservation happens after the choice is known, so an error path reserves nothing.
- Only `Drop for RouteGuard` releases a slot; it uses `saturating_sub` and a bounds-checked `get_mut`, so it cannot panic — an early return or a panic unwind anywhere in the request cannot leak capacity.
- Counters are process-local and deliberately not persisted: after a restart no request is in flight.

### Composition with `runtime::ModelAllocator`

- `runtime::allocator::ModelAllocator` runs **once per member at build time**, assigns one pool entry, and persists it as `MemberRecord::model_ref` (`model_name` + pool index); a member's model is stable across a conversation and survives a restart by rehydration.
- `ModelRouter` runs **once per request** over the same pool to spread load, honour an explicit request, or apply team rules; it persists nothing.
- Neither replaces the other: the allocator stops every member from piling onto one endpoint, the router keeps one busy member's requests spread (or pinned) and says which endpoint to retry.
- A team with an empty pool uses neither path; members keep the model their own spec declares.

### Pool contract

- `ModelPool::new` preserves declaration order exactly and rejects a blank (`trim().is_empty()`) `model_name` naming the index, and a duplicate `model_name` naming the name and both indices; both are `Error::validation`.
- The pool keeps a `HashMap` name index only for lookup; because duplicates are rejected, a name maps to exactly one position and `find` cannot be ambiguous.
- `model_group_label` deviates from Python deliberately: Python returns `None` for a non-model-group pool, Rust returns `String` — the IntelliRouter label (`metadata.model_group.display_name`, else `metadata.client.intelli_router.model_group_id`, else `"*"`) for a single `"*"`/`intelli_router` entry, otherwise the distinct model names joined by `", "` (empty string for an empty pool).

## Test Strategy

44 inline tests, deterministic and offline (`pool.rs`: 9 inline, `router/tests.rs`: 35 in the `router::tests` child module).
Pool (`pool.rs`): `pool_keeps_declaration_order_and_indexes_names`, `duplicate_model_names_are_rejected`, `blank_model_name_is_rejected_with_the_pool_index`, `empty_pool_is_allowed`, `from_entries_copies_the_slice`, `model_group_label_joins_distinct_model_names_in_pool_order`, `model_group_label_names_an_intelli_router_group`, `model_group_label_needs_a_single_entry`, `debug_lists_names_but_not_credentials`.
Router (`router.rs`): `pool_strategy_maps_onto_router_strategy`, `router_wire_names_match_the_documented_strings`, `route_reason_wire_names_match_the_documented_strings`, `complexity_is_ordered_and_serializes_snake_case`, `request_hint_builders_fill_the_expected_fields`, `request_hint_tolerates_unknown_json_fields`, `default_rule_matcher_matches_everything`, `rule_matcher_checks_role_and_task_kind`, `rule_matcher_checks_complexity_exactly_and_at_least`, `rule_matcher_checks_needs_tools_in_both_directions`, `routing_rule_reports_its_matcher_and_priority`, `router_debug_reports_shape_without_credentials`, `empty_pool_rejects_every_decision`, `for_pool_accepts_an_empty_slice_and_validates_entries`, `by_name_serves_the_requested_model`, `by_name_falls_back_to_the_first_entry`, `by_name_acquire_reserves_and_releases_the_slot`, `round_robin_preview_does_not_advance_the_cursor`, `round_robin_acquire_is_fair_over_the_pool`, `round_robin_ignores_the_requested_name`, `least_busy_spreads_over_the_pool_and_prefers_the_lowest_index`, `least_busy_fallback_chain_follows_current_load`, `least_busy_spreads_eight_concurrent_acquisitions`, `rule_based_applies_the_first_matching_rule`, `rule_based_orders_by_priority_then_name`, `rule_target_missing_from_the_pool_skips_to_the_next_rule`, `rule_based_falls_through_to_by_name_then_the_first_entry`, `rule_based_without_rules_uses_the_by_name_fallback`, `fallback_chain_is_deduplicated_and_excludes_the_chosen_model`, `fallback_chain_of_a_single_entry_pool_is_empty`, `guard_releases_its_slot_when_a_scope_returns_an_error`, `guard_releases_its_slot_on_panic_unwind`, `decision_round_trips_through_json`, `routing_rule_round_trips_through_json`, `rules_attached_to_a_non_rule_strategy_are_reported_but_unused`.
The concurrency test spawns 8 `tokio` tasks (multi-thread flavour) released by a `tokio::sync::Barrier`, holds every `RouteGuard` until all acquisitions finish, and asserts the exact per-index counts `[3, 3, 2]`; no sleeps and no wall-clock ordering.
Two doc examples (`mod.rs`, `router.rs`) also run under `cargo test --doc`.
Definition of done (all green): `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --offline --lib` (also `--no-default-features` and `--features otel`), `cargo clippy -p openjiuwen-agent-teams --all-targets --offline -- -D warnings` (also `--features otel`), and `cargo doc -p openjiuwen-agent-teams --no-deps` warning-free.

## Routing Table

- `./mod.rs` → module map, composition note with `runtime::ModelAllocator`, flat re-exports
- `./pool.rs` → `ModelPool`, validation, lookup, `model_group_label`
- `./router.rs` → `RouterStrategy`, `Complexity`, `RequestHint`, `RuleMatcher`, `RoutingRule`, `RouteReason`, `RouterDecision`, `ModelRouter`, `RouteGuard`
- `./router/tests.rs` → the 35 behavioural router tests (a child module of `router.rs`, split out to keep both files well under the 1000-line baseline)

## Deferred

- The IntelliRouter client-side reliable router and route-group service (`IntelliRouterConfig`, `IntelliRouterDeployment`, `INTELLI_ROUTER_PROVIDER`, `INTELLI_ROUTER_UNIFIED_MODEL` as public API, `RouterAllocator`, `IntelliRouterAllocator`, `CliModelCatalog`): needs the vendor router SDK/client and its deployment-probing loop, which this port does not carry; the model-group *label* is kept because it is display-only.
- `ModelPoolStrategy::IntelliRouter` itself is modelled: its `intelli_router` wire value round-trips and `src/runtime/allocator.rs` refuses it at allocation with a typed `Error::Unsupported`, so the deferral is a documented refusal rather than a parse failure.
- This module's `From<ModelPoolStrategy> for RouterStrategy` sends `IntelliRouter` to `RouterStrategy::RoundRobin`, which is safe because that conversion is never the path that silently degrades a team: an IntelliRouter team fails earlier, at allocation.
- Vendor CLI model catalogs (`resolve_cli_models`, `CliModelAllocation`): requires invoking the vendor `claude`/`codex` binaries, whose harness providers are deferred in `openjiuwen-harness-providers`.
- `ModelRouterConfig` expansion (one endpoint serving many names) and `inherit_pool_ids` (bit-exact `model_id` carry-over on pool refresh): both are spec-build/pool-refresh concerns of `schema`/`runtime`, not request routing; the pool here validates and preserves entries only.
- Weighted, LRU and affinity/KV-cache strategies: no persisted weight or affinity model exists in this module (cache identity and its registry live in `../kv_cache/`), so the four shipped strategies are the full surface.
- Persisting router state (`cursor`, in-flight counters) across restarts: deliberately not persisted — like Python, only the allocator's cursor is session state, and in-flight counts are meaningless after a restart.
